"""Orphaned-node fraction (no incoming layer-0 edge) across the separation sweep,
plus the uniform and digits cases. Usage: python experiments/exp_orphans.py"""
import json, sys
import numpy as np
from sklearn.datasets import load_digits
from common import build, gaussian, uniform, reachable_all_layers

def orphan_fraction(idx):
    indeg = dict.fromkeys(idx.vectors, 0)
    for u, nbrs in idx.layers[0].items():
        for v in nbrs:
            indeg[v] += 1
    return sum(1 for d in indeg.values() if d == 0) / len(indeg)

out = open("results/orphans.jsonl", "w")
cases = [("uniform", None)] + [("clusters", s) for s in (3.0, 2.0, 1.0, 0.5, 0.3)] + [("digits", None)]
for kind, std in cases:
    for seed in range(3):
        if kind == "uniform": X, _ = uniform(2000, 32, seed=seed)
        elif kind == "digits": X = load_digits().data.astype(float)
        else: X, _ = gaussian(2000, 32, 20, std, seed=seed)
        for rule in ("simple", "heuristic"):
            idx, _ = build(rule, X, seed=seed + 1)
            row = {"data": kind, "std": std, "seed": seed, "rule": rule, "orphans": orphan_fraction(idx), "reach_all": reachable_all_layers(idx)}
            print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()
