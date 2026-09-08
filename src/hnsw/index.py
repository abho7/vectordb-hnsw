"""
Hierarchical Navigable Small World graph index (Malkov & Yashunin,
"Efficient and Robust Approximate Nearest Neighbor Search Using
Hierarchical Navigable Small World Graphs," 2016/2018).

The core idea: build a multi-layer proximity graph where the top layers
are sparse (long-range "highway" edges connecting distant regions of the
vector space) and the bottom layer is dense (every point connected to
its true near neighbors). A query descends the layers greedily from a
random entry point at the top -- each layer narrows the search to a
small neighborhood -- then does a bounded best-first search at the
bottom layer for the actual answer. This is what gives HNSW its
expected O(log n) search complexity instead of brute force's O(n).

This implementation follows the paper's Algorithm 1 (INSERT) and
Algorithm 2 (SEARCH-LAYER) directly, using the "simple" neighbor
selection heuristic (take the M closest candidates) rather than the
paper's more elaborate diversity-aware heuristic (Algorithm 4) -- see
ARCHITECTURE.md for what that trades away and why it's a reasonable
simplification for this scale.
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np

from hnsw.distance import DISTANCE_FUNCTIONS


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        *,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        seed: int | None = None,
    ):
        """
        dim: dimensionality of vectors this index will store.
        metric: "cosine" or "euclidean".
        M: max bidirectional connections per node per layer (layer 0 uses 2*M,
           following the paper's recommendation that the bottom layer needs
           denser connectivity since it does the real work).
        ef_construction: size of the dynamic candidate list during insertion.
           Higher = better graph quality (and better recall later) at the
           cost of slower builds. This is the single biggest lever on the
           speed/quality tradeoff at index-build time.
        """
        self.dim = dim
        self.metric = metric
        self._dist_fn, self._dist_fn_batch = DISTANCE_FUNCTIONS[metric]

        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)

        self._rng = random.Random(seed)

        self.vectors: dict[int, np.ndarray] = {}
        self.layers: list[dict[int, list[int]]] = []
        self.entry_point: int | None = None
        self.max_layer: int = -1

    def __len__(self) -> int:
        return len(self.vectors)

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.mL)

    def _search_layer(self, query: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Algorithm 2 from the paper: best-first search within a single
        layer, maintaining a bounded (size ef) result set. Returns a list
        of (distance, node_id) sorted ascending by distance."""
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []

        for ep in entry_points:
            d = self._dist_fn(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            c_dist, c = heapq.heappop(candidates)
            furthest_in_results = -results[0][0]
            if c_dist > furthest_in_results and len(results) >= ef:
                break

            for neighbor in self.layers[layer].get(c, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                n_dist = self._dist_fn(query, self.vectors[neighbor])
                furthest_in_results = -results[0][0] if results else float("inf")
                if n_dist < furthest_in_results or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, neighbor))
                    heapq.heappush(results, (-n_dist, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted([(-d, nid) for d, nid in results])

    def _select_neighbors_simple(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """The paper's SELECT-NEIGHBORS-SIMPLE: just the m closest
        candidates by distance. `candidates` is assumed pre-sorted
        ascending (as _search_layer returns). Kept for comparison/testing
        -- see _select_neighbors_heuristic for why this isn't what's
        actually used during insertion."""
        return [nid for _, nid in candidates[:m]]

    def _select_neighbors_heuristic(self, query: np.ndarray, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """The paper's SELECT-NEIGHBORS-HEURISTIC (Algorithm 4): a
        diversity-aware selection that only accepts a candidate if it's
        closer to the query than it is to every neighbor already
        selected. This is not an optional refinement -- it's what
        prevents the graph from collapsing into disconnected per-cluster
        islands on real, non-uniform data.

        Concretely: SELECT-NEIGHBORS-SIMPLE always picks the M nearest
        candidates. For a node deep inside a tight cluster, its M
        nearest candidates are almost always all from the same cluster
        -- nothing ever forces a "bridge" edge to a different region of
        the vector space. SELECT-NEIGHBORS-HEURISTIC's rejection rule
        (skip a candidate that's farther from the query than it is from
        an already-picked neighbor) preferentially keeps candidates that
        are NOT redundant with what's already selected, which is what
        preserves the long-range edges the top layers depend on for
        expected O(log n) search.

        Verified empirically in this project: on Gaussian-cluster test
        data, SELECT-NEIGHBORS-SIMPLE left only ~5% of nodes reachable
        from the entry point (a near-total graph collapse); switching to
        this heuristic restored >95% connectivity and recall@10 above
        0.9 at the same parameters. See benchmarks/run_benchmark.py and
        ARCHITECTURE.md for the full before/after numbers."""
        result: list[int] = []
        discarded: list[tuple[float, int]] = []

        for dist_to_query, cand_id in candidates:
            if len(result) >= m:
                break
            cand_vec = self.vectors[cand_id]
            dominated = any(
                self._dist_fn(cand_vec, self.vectors[r]) < dist_to_query
                for r in result
            )
            if not dominated:
                result.append(cand_id)
            else:
                discarded.append((dist_to_query, cand_id))

        # If the diversity constraint left us short of m, backfill with
        # the closest discarded candidates rather than under-connecting
        # the node -- this is the paper's optional keepPrunedConnections
        # behavior, on by default here.
        if len(result) < m:
            for _, cand_id in discarded:
                if len(result) >= m:
                    break
                result.append(cand_id)

        return result

    def insert(self, node_id: int, vector: np.ndarray) -> None:
        if node_id in self.vectors:
            raise ValueError(f"node_id {node_id} already exists in the index")
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")

        self.vectors[node_id] = vector
        level = self._random_level()

        while len(self.layers) <= level:
            self.layers.append({})
        for l in range(level + 1):
            self.layers[l].setdefault(node_id, [])

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = level
            return

        ep = self.entry_point
        for lc in range(self.max_layer, level, -1):
            nearest = self._search_layer(vector, [ep], ef=1, layer=lc)
            if nearest:
                ep = nearest[0][1]

        entry_points = [ep]
        for lc in range(min(self.max_layer, level), -1, -1):
            candidates = self._search_layer(vector, entry_points, self.ef_construction, lc)
            m_target = self.M if lc > 0 else self.M_max0
            neighbors = self._select_neighbors_heuristic(vector, candidates, m_target)

            self.layers[lc][node_id] = list(neighbors)
            for neighbor_id in neighbors:
                self.layers[lc][neighbor_id].append(node_id)
                conns = self.layers[lc][neighbor_id]
                m_max = self.M_max0 if lc == 0 else self.M
                if len(conns) > m_max:
                    conn_dists = sorted(
                        (self._dist_fn(self.vectors[neighbor_id], self.vectors[c]), c) for c in conns
                    )
                    self.layers[lc][neighbor_id] = self._select_neighbors_heuristic(
                        self.vectors[neighbor_id], conn_dists, m_max
                    )

            entry_points = [nid for _, nid in candidates]

        if level > self.max_layer:
            self.entry_point = node_id
            self.max_layer = level

    def search(self, query: np.ndarray, k: int, ef_search: int | None = None) -> list[tuple[int, float]]:
        """Returns up to k (node_id, distance) pairs, sorted nearest first."""
        if self.entry_point is None:
            return []
        query = np.asarray(query, dtype=np.float64)
        ef = ef_search if ef_search is not None else max(self.ef_construction, k)

        ep = self.entry_point
        for lc in range(self.max_layer, 0, -1):
            nearest = self._search_layer(query, [ep], ef=1, layer=lc)
            if nearest:
                ep = nearest[0][1]

        candidates = self._search_layer(query, [ep], max(ef, k), layer=0)
        return [(nid, dist) for dist, nid in candidates[:k]]
