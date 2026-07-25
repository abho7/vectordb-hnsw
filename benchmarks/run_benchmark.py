#!/usr/bin/env python3
"""
Benchmarks HNSW against brute-force exact search: recall@k (what
fraction of the true k nearest neighbors HNSW actually finds) and query
latency, across dataset sizes. Run with: python benchmarks/run_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from datasets import gaussian_clusters
from hnsw.index import HNSWIndex


def brute_force_knn(vectors: dict[int, np.ndarray], query: np.ndarray, k: int, dist_fn) -> list[int]:
    scored = sorted(((dist_fn(v, query), i) for i, v in vectors.items()))
    return [i for _, i in scored[:k]]


def recall_at_k(hnsw_ids: list[int], true_ids: list[int]) -> float:
    if not true_ids:
        return 1.0
    return len(set(hnsw_ids) & set(true_ids)) / len(true_ids)


def run_benchmark(n: int, dim: int, k: int, n_queries: int, ef_search: int) -> dict:
    vectors = gaussian_clusters(n, dim, n_clusters=max(5, n // 100), seed=42)

    index = HNSWIndex(dim=dim, metric="euclidean", M=16, ef_construction=200, seed=1)
    build_start = time.perf_counter()
    for i, v in vectors.items():
        index.insert(i, v)
    build_time = time.perf_counter() - build_start

    from hnsw.distance import euclidean
    rng = np.random.default_rng(123)
    query_ids = rng.choice(list(vectors.keys()), size=min(n_queries, n), replace=False)

    recalls = []
    hnsw_times = []
    brute_times = []
    for qid in query_ids:
        query = vectors[qid]

        t0 = time.perf_counter()
        hnsw_results = index.search(query, k=k, ef_search=ef_search)
        hnsw_times.append(time.perf_counter() - t0)
        hnsw_ids = [nid for nid, _ in hnsw_results]

        t0 = time.perf_counter()
        true_ids = brute_force_knn(vectors, query, k, euclidean)
        brute_times.append(time.perf_counter() - t0)

        recalls.append(recall_at_k(hnsw_ids, true_ids))

    return {
        "n": n,
        "dim": dim,
        "k": k,
        "build_time_s": build_time,
        "avg_recall": sum(recalls) / len(recalls),
        "avg_hnsw_query_ms": sum(hnsw_times) / len(hnsw_times) * 1000,
        "avg_brute_query_ms": sum(brute_times) / len(brute_times) * 1000,
        "speedup": (sum(brute_times) / len(brute_times)) / (sum(hnsw_times) / len(hnsw_times)),
    }


def main():
    print(f"{'N':>8} {'dim':>5} {'build(s)':>10} {'recall@10':>10} {'HNSW(ms)':>10} {'brute(ms)':>11} {'speedup':>9}")
    print("-" * 70)
    for n in [500, 2000, 8000]:
        result = run_benchmark(n=n, dim=32, k=10, n_queries=50, ef_search=100)
        print(
            f"{result['n']:>8} {result['dim']:>5} {result['build_time_s']:>10.3f} "
            f"{result['avg_recall']:>10.3f} {result['avg_hnsw_query_ms']:>10.4f} "
            f"{result['avg_brute_query_ms']:>11.4f} {result['speedup']:>8.1f}x"
        )


if __name__ == "__main__":
    main()
