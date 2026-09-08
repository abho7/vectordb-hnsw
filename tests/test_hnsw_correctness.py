from collections import deque

import numpy as np

from hnsw.distance import euclidean
from hnsw.index import HNSWIndex


def _brute_force_knn(vectors: dict[int, np.ndarray], query: np.ndarray, k: int) -> list[int]:
    scored = sorted(((euclidean(v, query), i) for i, v in vectors.items()))
    return [i for _, i in scored[:k]]


def _reachable_from_entry_point(index: HNSWIndex) -> set[int]:
    if index.entry_point is None:
        return set()
    visited = {index.entry_point}
    q = deque([index.entry_point])
    while q:
        node = q.popleft()
        for neighbor in index.layers[0].get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return visited


def test_exact_match_recall_on_small_uniform_dataset():
    rng = np.random.default_rng(0)
    index = HNSWIndex(dim=8, metric="euclidean", M=8, ef_construction=100, seed=1)
    vectors = {i: rng.normal(size=8) for i in range(200)}
    for i, v in vectors.items():
        index.insert(i, v)

    query = rng.normal(size=8)
    true_ids = set(_brute_force_knn(vectors, query, 10))
    hnsw_ids = set(nid for nid, _ in index.search(query, k=10, ef_search=100))
    assert hnsw_ids == true_ids


def test_graph_remains_connected_on_clustered_data():
    """Regression test for a real bug found during development: the
    'simple' neighbor-selection heuristic (just take the M nearest
    candidates) causes the layer-0 graph to collapse into disconnected
    per-cluster islands on well-separated data -- only ~5% of nodes were
    reachable from the entry point before this was fixed by switching to
    the paper's diversity-aware SELECT-NEIGHBORS-HEURISTIC. This test
    locks in that fix: connectivity must stay high even on adversarially
    clustered data, not just uniform random data where the bug was
    invisible."""
    rng = np.random.default_rng(42)
    n_clusters, n_per_cluster, dim = 15, 40, 16
    centers = rng.normal(scale=4.0, size=(n_clusters, dim))

    index = HNSWIndex(dim=dim, metric="euclidean", M=16, ef_construction=150, seed=1)
    node_id = 0
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            v = centers[c] + rng.normal(scale=0.3, size=dim)
            index.insert(node_id, v)
            node_id += 1

    reachable = _reachable_from_entry_point(index)
    total = n_clusters * n_per_cluster
    assert len(reachable) / total > 0.95, (
        f"only {len(reachable)}/{total} nodes reachable from entry point -- "
        "graph has collapsed into disconnected clusters"
    )


def test_recall_at_10_above_90_percent_on_clustered_data():
    rng = np.random.default_rng(7)
    n_clusters, n_per_cluster, dim = 10, 50, 24
    centers = rng.normal(scale=4.0, size=(n_clusters, dim))

    vectors = {}
    node_id = 0
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            vectors[node_id] = centers[c] + rng.normal(scale=0.3, size=dim)
            node_id += 1

    index = HNSWIndex(dim=dim, metric="euclidean", M=16, ef_construction=150, seed=1)
    for i, v in vectors.items():
        index.insert(i, v)

    query_ids = rng.choice(list(vectors.keys()), size=25, replace=False)
    recalls = []
    for qid in query_ids:
        q = vectors[qid]
        true_ids = set(_brute_force_knn(vectors, q, 10))
        hnsw_ids = set(nid for nid, _ in index.search(q, k=10, ef_search=100))
        recalls.append(len(true_ids & hnsw_ids) / 10)

    avg_recall = sum(recalls) / len(recalls)
    assert avg_recall > 0.9, f"recall@10 = {avg_recall:.3f}, expected > 0.9"


def test_empty_index_search_returns_empty():
    index = HNSWIndex(dim=4)
    assert index.search(np.zeros(4), k=5) == []


def test_single_element_index():
    index = HNSWIndex(dim=4)
    index.insert(0, np.array([1.0, 2.0, 3.0, 4.0]))
    results = index.search(np.array([1.0, 2.0, 3.0, 4.0]), k=5)
    assert len(results) == 1
    assert results[0][0] == 0


def test_duplicate_node_id_raises():
    index = HNSWIndex(dim=4)
    index.insert(0, np.zeros(4))
    try:
        index.insert(0, np.ones(4))
        assert False, "expected ValueError for duplicate node_id"
    except ValueError:
        pass


def test_wrong_dimension_raises():
    index = HNSWIndex(dim=4)
    try:
        index.insert(0, np.zeros(3))
        assert False, "expected ValueError for wrong dimension"
    except ValueError:
        pass


def test_k_larger_than_index_size_returns_all():
    index = HNSWIndex(dim=4)
    for i in range(3):
        index.insert(i, np.random.default_rng(i).normal(size=4))
    results = index.search(np.zeros(4), k=100)
    assert len(results) == 3
