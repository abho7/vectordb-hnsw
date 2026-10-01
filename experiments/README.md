# Experiments

Scripts behind the ICCE 2027 paper "Diversity-Aware Neighbor Selection Prevents
Graph Connectivity Collapse in HNSW Vector Search for On-Device Semantic Retrieval".
Raw outputs are in `../results/`. Nothing in `src/` is modified: the simple
neighbor-selection rule is a subclass in `common.py` that overrides one method.

Run every script from the repository root.

| Script | What it measures |
|---|---|
| `exp_readme_repro.py simple heuristic` | The original collapse configuration (n=2000, 20 clusters, std 0.3): reachability, recall@10 at ef 10/100/500/2000. |
| `exp_connectivity.py` | Separation sweep (uniform + std 3, 2, 1, 0.5, 0.3; 3 seeds): reachability and recall for both rules. |
| `separation.py` | Cluster-separation ratio S for each synthetic setting and for the UCI digits (run from `experiments/`). |
| `exp_orphans.py` | Fraction of nodes with no incoming layer-0 edge, for every configuration. |
| `exp_real_digits.py` | Both rules on sklearn's bundled UCI handwritten digits (1,797 x 64). |
| `exp_scaling.py` | Heuristic index at n = 500, 2000, 8000: recall, distance computations per query, latency vs a Python-loop scan and a vectorized NumPy scan, graph size. |

Requires `numpy` and `scikit-learn` (for the digits dataset). Latencies depend on the machine.
