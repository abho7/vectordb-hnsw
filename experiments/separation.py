"""Cluster separation ratio S = mean distance between class centroids /
mean distance between two random points of the same class. Used to place
synthetic and real datasets on one axis."""
import numpy as np
from sklearn.datasets import load_digits
from common import gaussian

def ratio(X, lab, rng=np.random.default_rng(0)):
    cls = np.unique(lab); C = np.array([X[lab == c].mean(0) for c in cls])
    inter = np.mean([np.linalg.norm(C[i] - C[j]) for i in range(len(C)) for j in range(i + 1, len(C))])
    intra = []
    for c in cls:
        P = X[lab == c]
        a, b = rng.integers(0, len(P), 200), rng.integers(0, len(P), 200)
        intra.append(np.linalg.norm(P[a] - P[b], axis=1).mean())
    return inter / np.mean(intra)

if __name__ == "__main__":
    d = load_digits(); print(f"digits S={ratio(d.data, d.target):.2f}")
    for std in (3.0, 2.0, 1.0, 0.5, 0.3):
        X, lab = gaussian(2000, 32, 20, std, seed=0); print(f"gaussian std={std} S={ratio(X, lab):.2f}")
