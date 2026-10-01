"""Shared helpers for the ICCE experiments. Nothing in src/ is modified."""
from __future__ import annotations
import sys, time
from collections import deque
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "benchmarks"))
from hnsw.index import HNSWIndex  # noqa: E402


class SimpleHNSW(HNSWIndex):
    """SELECT-NEIGHBORS-SIMPLE everywhere the heuristic is used (selection and pruning)."""
    def _select_neighbors_heuristic(self, query, candidates, m):
        return [nid for _, nid in candidates[:m]]


RULES = {"heuristic": HNSWIndex, "simple": SimpleHNSW}


class Counted:
    """Wrap an index's distance function to count evaluations."""
    def __init__(self, index):
        self.index, self.n = index, 0
        f = index._dist_fn
        def g(a, b):
            self.n += 1
            return f(a, b)
        index._dist_fn = g


def build(rule: str, X: np.ndarray, metric="euclidean", M=16, efc=200, seed=1):
    idx = RULES[rule](dim=X.shape[1], metric=metric, M=M, ef_construction=efc, seed=seed)
    t0 = time.perf_counter()
    for i, v in enumerate(X):
        idx.insert(i, v)
    return idx, time.perf_counter() - t0


def reachable_fraction(idx) -> float:
    """Fraction of nodes reachable from the entry point over layer-0 edges."""
    g = idx.layers[0]
    seen, dq = {idx.entry_point}, deque([idx.entry_point])
    while dq:
        u = dq.popleft()
        for v in g.get(u, []):
            if v not in seen:
                seen.add(v); dq.append(v)
    return len(seen) / len(idx.vectors)


def inter_cluster_edge_fraction(idx, labels) -> float:
    g = idx.layers[0]
    tot = cross = 0
    for u, nbrs in g.items():
        for v in nbrs:
            tot += 1; cross += labels[u] != labels[v]
    return cross / max(tot, 1)


def exact_knn(X: np.ndarray, q: np.ndarray, k: int, metric="euclidean"):
    if metric == "euclidean":
        d = np.linalg.norm(X - q, axis=1)
    else:
        n = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
        d = 1 - (X @ q) / np.where(n == 0, 1, n)
    return set(np.argsort(d, kind="stable")[:k].tolist())


def recall(idx, X, qids, k=10, ef=100, metric="euclidean"):
    rs = []
    for qi in qids:
        got = {nid for nid, _ in idx.search(X[qi], k=k, ef_search=ef)}
        rs.append(len(got & exact_knn(X, X[qi], k, metric)) / k)
    return float(np.mean(rs))


def gaussian(n, dim, n_clusters, std, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(n_clusters, dim))
    lab = rng.integers(0, n_clusters, size=n)
    return centers[lab] + rng.normal(scale=std, size=(n, dim)), lab


def uniform(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 3.0, size=(n, dim)), np.zeros(n, dtype=int)


def reachable_all_layers(idx) -> float:
    """Fraction of nodes reachable from the entry point using edges of ANY
    layer (a search can traverse upper layers, then descend)."""
    seen, dq = {idx.entry_point}, deque([idx.entry_point])
    while dq:
        u = dq.popleft()
        for layer in idx.layers:
            for v in layer.get(u, []):
                if v not in seen:
                    seen.add(v); dq.append(v)
    return len(seen) / len(idx.vectors)
