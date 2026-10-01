"""E4/E5: scaling of the (heuristic) index on the README's clustered data:
recall@10, distance evaluations per query, latency vs. two brute-force
baselines (the repo's Python loop and a vectorized NumPy scan), graph size."""
import json, sys, time
import numpy as np
from common import build, recall, Counted, exact_knn
sys.path.insert(0, "benchmarks")
from datasets import gaussian_clusters
from hnsw.distance import euclidean

def loop_brute(V, q, k):
    return [i for _, i in sorted((euclidean(v, q), i) for i, v in V.items())[:k]]

with open("results/scaling.jsonl", "w") as f:
    for n in (500, 2000, 8000):
        V = gaussian_clusters(n, 32, n_clusters=max(5, n // 100), seed=42)
        X = np.array([V[i] for i in range(n)])
        idx, bt = build("heuristic", X, seed=1)
        qids = np.random.default_rng(123).choice(n, 50, replace=False)
        rec = recall(idx, X, qids, ef=100)
        c = Counted(idx); t_h = []
        for qi in qids:
            t0 = time.perf_counter(); idx.search(X[qi], k=10, ef_search=100); t_h.append(time.perf_counter() - t0)
        evals = c.n / len(qids)
        t_loop = []; t_np = []
        for qi in qids:
            t0 = time.perf_counter(); loop_brute(V, X[qi], 10); t_loop.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); exact_knn(X, X[qi], 10); t_np.append(time.perf_counter() - t0)
        edges0 = sum(len(v) for v in idx.layers[0].values())
        edges_all = sum(len(v) for L in idx.layers for v in L.values())
        row = {"n": n, "build_s": bt, "recall_ef100": rec, "dist_evals_per_query": evals,
               "evals_frac_of_n": evals / n, "hnsw_ms": 1000 * np.median(t_h), "loop_brute_ms": 1000 * np.median(t_loop),
               "numpy_brute_ms": 1000 * np.median(t_np), "avg_degree_l0": edges0 / n, "edges_all": edges_all,
               "graph_bytes_per_vec_int32": 4 * edges_all / n, "vector_bytes_float32": 4 * 32, "layers": len(idx.layers)}
        print(json.dumps(row), flush=True); f.write(json.dumps(row) + "\n")
