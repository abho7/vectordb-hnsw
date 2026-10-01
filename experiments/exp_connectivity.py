"""E1/E2: connectivity and recall of SIMPLE vs HEURISTIC neighbor selection
as cluster separation varies. Usage: python experiments/exp_connectivity.py [--n 1000]"""
import argparse, json
import numpy as np
from common import build, reachable_fraction, reachable_all_layers, inter_cluster_edge_fraction, recall, gaussian, uniform

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--dim", type=int, default=32); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default="results/connectivity.jsonl")
a = ap.parse_args()
settings = [("uniform", None)] + [("clusters", s) for s in (3.0, 2.0, 1.0, 0.5, 0.3)]
with open(a.out, "w") as f:
    for seed in range(a.seeds):
        for kind, std in settings:
            X, lab = uniform(a.n, a.dim, seed=seed) if kind == "uniform" else gaussian(a.n, a.dim, 20, std, seed=seed)
            qids = np.random.default_rng(100 + seed).choice(a.n, 50, replace=False)
            for rule in ("simple", "heuristic"):
                idx, bt = build(rule, X, seed=seed + 1)
                row = {"seed": seed, "data": kind, "std": std, "rule": rule, "n": a.n,
                       "reach_l0": reachable_fraction(idx), "reach_all": reachable_all_layers(idx),
                       "cross_edges": inter_cluster_edge_fraction(idx, lab) if kind != "uniform" else None,
                       "recall_ef100": recall(idx, X, qids, ef=100),
                       "recall_efN": recall(idx, X, qids[:20], ef=a.n), "build_s": bt}
                print(json.dumps(row), flush=True); f.write(json.dumps(row) + "\n"); f.flush()
