"""
Distance functions used by the HNSW index. Kept separate from the index
itself so the metric is swappable without touching graph-construction
logic.

All functions take numpy arrays and are vectorized where the caller
passes a batch (a 2D array) instead of a single vector, since batched
distance computation is what most of the wall-clock time in brute-force
comparison and neighbor-selection actually goes to.
"""

from __future__ import annotations

import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def euclidean_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """query: (d,), candidates: (n, d) -> (n,) distances."""
    return np.linalg.norm(candidates - query, axis=1)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def cosine_distance_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    cand_norms = np.linalg.norm(candidates, axis=1)
    denom = query_norm * cand_norms
    dots = candidates @ query
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > 0, dots / denom, 0.0)
    return 1.0 - sims


DISTANCE_FUNCTIONS = {
    "euclidean": (euclidean, euclidean_batch),
    "cosine": (cosine_distance, cosine_distance_batch),
}
