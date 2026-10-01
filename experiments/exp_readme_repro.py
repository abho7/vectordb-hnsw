"""Reproduce the README/ARCHITECTURE configuration exactly: n=2000, dim=32,
20 Gaussian clusters, std=0.3, data seed 42, M=16, efC=200, index seed 1."""
import json, sys
import numpy as np
from common import build, reachable_fraction, reachable_all_layers, recall
sys.path.insert(0, "benchmarks")
from datasets import gaussian_clusters
V = gaussian_clusters(2000, 32, n_clusters=20, seed=42)
X = np.array([V[i] for i in range(2000)])
qids = np.random.default_rng(123).choice(2000, 50, replace=False)
for rule in sys.argv[1:]:
    idx, bt = build(rule, X, seed=1)
    row = {"rule": rule, "reach_l0": reachable_fraction(idx), "reach_all": reachable_all_layers(idx), "build_s": bt}
    for ef in (10, 100, 500, 2000):
        row[f"recall_ef{ef}"] = recall(idx, X, qids, ef=ef)
    print(json.dumps(row), flush=True)
