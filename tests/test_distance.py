import numpy as np

from hnsw.distance import cosine_distance, cosine_distance_batch, euclidean, euclidean_batch


def test_euclidean_identical_vectors_zero_distance():
    v = np.array([1.0, 2.0, 3.0])
    assert euclidean(v, v) == 0.0


def test_euclidean_known_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean(a, b) == 5.0


def test_cosine_identical_direction_zero_distance():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_distance(v, v)) < 1e-9


def test_cosine_orthogonal_vectors_distance_one():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_distance(a, b) - 1.0) < 1e-9


def test_cosine_opposite_vectors_distance_two():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_distance(a, b) - 2.0) < 1e-9


def test_cosine_scale_invariant():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    assert abs(cosine_distance(a, b)) < 1e-9


def test_batch_matches_single_computation():
    rng = np.random.default_rng(0)
    query = rng.normal(size=16)
    candidates = rng.normal(size=(20, 16))

    batch_result = euclidean_batch(query, candidates)
    for i in range(20):
        assert abs(batch_result[i] - euclidean(query, candidates[i])) < 1e-9

    batch_cos = cosine_distance_batch(query, candidates)
    for i in range(20):
        assert abs(batch_cos[i] - cosine_distance(query, candidates[i])) < 1e-9


def test_zero_vector_cosine_distance_does_not_crash():
    zero = np.zeros(4)
    other = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_distance(zero, other) == 1.0
