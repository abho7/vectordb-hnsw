"""E3: real data. sklearn's bundled UCI handwritten-digits images (1,797 x 64,
10 natural classes), euclidean. Compares SIMPLE vs HEURISTIC selection."""
import json
import numpy as np
from sklearn.datasets import load_digits
from common import build, reachable_fraction, reachable_all_layers, inter_cluster_edge_fraction, recall
d = load_digits(); X = d.data.astype(np.float64); lab = d.target
qids = np.random.default_rng(7).choice(len(X), 100, replace=False)
with open("results/digits.jsonl", "w") as f:
    for seed in range(3):
        for rule in ("simple", "heuristic"):
            idx, bt = build(rule, X, seed=seed + 1)
            row = {"seed": seed, "rule": rule, "reach_l0": reachable_fraction(idx), "reach_all": reachable_all_layers(idx),
                   "cross_edges": inter_cluster_edge_fraction(idx, lab), "build_s": bt}
            for ef in (10, 50, 100):
                row[f"recall_ef{ef}"] = recall(idx, X, qids, ef=ef)
            print(json.dumps(row), flush=True); f.write(json.dumps(row) + "\n")
