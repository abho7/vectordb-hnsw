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
Algorithm 2 (SEARCH-LAYER) directly, and selects neighbors with the
paper's diversity-aware SELECT-NEIGHBORS-HEURISTIC (Algorithm 4)
rather than the "simple" rule of taking the M closest candidates --
see ARCHITECTURE.md for what the simple rule costs: on clustered data
it left only 4.8% of nodes reachable from the entry point, and recall
stuck at 0.44 no matter how large ef_search was set.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Mapping

import numpy as np

from hnsw.distance import DISTANCE_FUNCTIONS

# Bumped whenever the array layout below changes in a way an older reader
# would misinterpret. A reader that finds a version it does not know must
# refuse the snapshot rather than guess at it.
SNAPSHOT_VERSION = 1


class SnapshotError(ValueError):
    """The arrays handed to HNSWIndex.from_arrays() are not a usable snapshot.

    A ValueError subclass for the same reason as RestoreError: callers already
    catching ValueError keep working, and a caller that would rather rebuild
    from source data than fail can catch this specifically.
    """


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

    # -- snapshots ---------------------------------------------------------
    #
    # The graph's on-disk shape is the engine's business, not its callers'.
    # Before this existed, the one consumer that persists an index wrote the
    # adjacency out itself and assigned straight back into .layers, .vectors,
    # .entry_point and .max_layer on the way in -- which meant a change to any
    # of those representations was a silent break out there rather than a
    # failure here. These two methods are the whole contract: arrays out,
    # arrays in, with a version stamp so a layout change is refused instead of
    # misread.
    #
    # Keys are namespaced so the dict can be merged into a caller's own
    # container -- an .npz alongside its own bookkeeping -- without collisions.

    def to_arrays(self) -> dict[str, np.ndarray]:
        """The graph as a flat dict of arrays, ready for any array container.

        Adjacency goes out in CSR form -- nodes, offsets, flat neighbours --
        so that a node with no neighbours stays distinguishable from a node
        that is absent entirely. Insertion relies on that distinction.

        Vectors are stored as float32 while the index computes in float64.
        For an index built from float32 data -- which is what every embedding
        model this is pointed at produces -- the round trip is exact. For one
        built from genuinely float64 vectors it is not: the values come back
        rounded, and distances shift in the last few digits, which can reorder
        two neighbours that were already tied. Said plainly rather than left
        to be discovered, because halving the file is worth it for the data
        this actually stores, and would not be otherwise.
        """
        internal_ids = sorted(self.vectors.keys())
        vectors = (
            np.stack([self.vectors[i] for i in internal_ids]).astype(np.float32)
            if internal_ids
            else np.zeros((0, self.dim), dtype=np.float32)
        )

        arrays: dict[str, np.ndarray] = {
            "hnsw_version": np.array(SNAPSHOT_VERSION, dtype=np.int64),
            "hnsw_dim": np.array(self.dim, dtype=np.int64),
            "hnsw_internal_ids": np.array(internal_ids, dtype=np.int64),
            "hnsw_vectors": vectors,
            # -1 rather than a missing key, so "no entry point" survives a
            # container that cannot store None.
            "hnsw_entry_point": np.array(
                -1 if self.entry_point is None else self.entry_point, dtype=np.int64
            ),
            "hnsw_max_layer": np.array(self.max_layer, dtype=np.int64),
            "hnsw_num_layers": np.array(len(self.layers), dtype=np.int64),
        }

        for layer_num, adjacency in enumerate(self.layers):
            nodes = sorted(adjacency.keys())
            offsets = [0]
            flat: list[int] = []
            for node in nodes:
                flat.extend(adjacency[node])
                offsets.append(len(flat))
            arrays[f"hnsw_L{layer_num}_nodes"] = np.array(nodes, dtype=np.int64)
            arrays[f"hnsw_L{layer_num}_offsets"] = np.array(offsets, dtype=np.int64)
            arrays[f"hnsw_L{layer_num}_neighbors"] = np.array(flat, dtype=np.int64)

        return arrays

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        seed: int | None = None,
    ) -> "HNSWIndex":
        """Rebuild a graph previously produced by to_arrays().

        Raises SnapshotError if the arrays are missing, malformed, of an
        unknown version, or describe a graph that does not hang together, so
        that a caller can fall back to rebuilding from source data rather than
        adopt an index that is quietly wrong.

        Build parameters come from the caller rather than from the snapshot:
        they govern how the graph grows from here, so the caller's current
        configuration should win over whatever was in force when the snapshot
        was taken. Only dim is checked, because a mismatch there means the
        vectors cannot be used at all.
        """

        def need(key: str) -> np.ndarray:
            try:
                return arrays[key]
            except (KeyError, TypeError) as exc:
                raise SnapshotError(f"snapshot is missing {key!r}") from exc

        try:
            version = int(need("hnsw_version"))
        except (ValueError, TypeError) as exc:
            raise SnapshotError("hnsw_version is not an integer") from exc
        if version != SNAPSHOT_VERSION:
            raise SnapshotError(
                f"snapshot version {version}, this engine reads {SNAPSHOT_VERSION}"
            )

        try:
            stored_dim = int(need("hnsw_dim"))
            internal_ids = [int(i) for i in np.asarray(need("hnsw_internal_ids")).tolist()]
            vectors = np.asarray(need("hnsw_vectors"))
            entry_point = int(need("hnsw_entry_point"))
            max_layer = int(need("hnsw_max_layer"))
            num_layers = int(need("hnsw_num_layers"))
        except (ValueError, TypeError) as exc:
            raise SnapshotError(f"snapshot is malformed: {exc}") from exc

        if stored_dim != dim:
            raise SnapshotError(f"snapshot dim {stored_dim} != requested dim {dim}")
        if vectors.ndim != 2 or vectors.shape[1] != dim:
            raise SnapshotError(
                f"snapshot vectors have shape {vectors.shape}, expected (n, {dim})"
            )
        if len(internal_ids) != len(vectors):
            raise SnapshotError(f"{len(internal_ids)} ids against {len(vectors)} vectors")
        if len(set(internal_ids)) != len(internal_ids):
            raise SnapshotError("snapshot has duplicate internal ids")
        if num_layers < 0:
            raise SnapshotError(f"negative layer count {num_layers}")

        index = cls(dim=dim, metric=metric, M=M, ef_construction=ef_construction, seed=seed)
        index.vectors = {
            node_id: np.asarray(vector, dtype=np.float64)
            for node_id, vector in zip(internal_ids, vectors)
        }

        known = set(index.vectors)
        layers: list[dict[int, list[int]]] = []
        for layer_num in range(num_layers):
            nodes = [int(n) for n in np.asarray(need(f"hnsw_L{layer_num}_nodes")).tolist()]
            offsets = [int(o) for o in np.asarray(need(f"hnsw_L{layer_num}_offsets")).tolist()]
            flat = [int(n) for n in np.asarray(need(f"hnsw_L{layer_num}_neighbors")).tolist()]

            if len(offsets) != len(nodes) + 1:
                raise SnapshotError(
                    f"layer {layer_num}: {len(nodes)} nodes against {len(offsets)} "
                    f"offsets, expected {len(nodes) + 1}"
                )
            if offsets[0] != 0 or offsets[-1] != len(flat):
                raise SnapshotError(f"layer {layer_num}: offsets do not span the neighbours")
            if any(b < a for a, b in zip(offsets, offsets[1:])):
                raise SnapshotError(f"layer {layer_num}: offsets are not monotonic")

            adjacency = {
                node: [int(n) for n in flat[offsets[i] : offsets[i + 1]]]
                for i, node in enumerate(nodes)
            }
            # A neighbour with no vector would be followed during search and
            # fail there instead of here, a long way from the cause.
            unknown = {n for neighbours in adjacency.values() for n in neighbours} - known
            if unknown:
                raise SnapshotError(
                    f"layer {layer_num} points at {len(unknown)} node(s) with no vector"
                )
            layers.append(adjacency)

        index.layers = layers
        index.entry_point = None if entry_point < 0 else entry_point
        index.max_layer = max_layer

        if index.entry_point is not None and index.entry_point not in known:
            raise SnapshotError(f"entry point {index.entry_point} has no vector")
        if known and index.entry_point is None:
            raise SnapshotError("snapshot has vectors but no entry point")
        if max_layer >= num_layers:
            raise SnapshotError(f"max_layer {max_layer} against {num_layers} layer(s)")

        return index
