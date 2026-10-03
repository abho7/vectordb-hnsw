"""Snapshot round trips: to_arrays/from_arrays, and the VectorDB wrappers.

The point of this interface is that a caller persisting an index never has to
model how the graph is laid out. These tests hold it to that: a round trip has
to return the *same answers*, not merely a structurally similar graph, and a
snapshot that does not hang together has to be refused rather than adopted.
"""

import numpy as np
import pytest

from hnsw.index import SNAPSHOT_VERSION, HNSWIndex, SnapshotError
from vectordb.store import RestoreError, VectorDB


def build_index(n=60, dim=8, seed=0):
    index = HNSWIndex(dim=dim, metric="euclidean", seed=1)
    rng = np.random.default_rng(seed)
    for node_id in range(n):
        index.insert(node_id, rng.normal(size=dim))
    return index


def test_round_trip_is_exact_for_float32_vectors():
    """The case that matters: every embedding model this is pointed at emits
    float32, and the snapshot stores float32, so nothing is lost."""
    index = HNSWIndex(dim=8, metric="euclidean", seed=1)
    rng = np.random.default_rng(0)
    for node_id in range(60):
        index.insert(node_id, rng.normal(size=8).astype(np.float32))

    query = rng.normal(size=8).astype(np.float32)
    before = index.search(query, k=10)

    restored = HNSWIndex.from_arrays(index.to_arrays(), dim=8, metric="euclidean")

    assert restored.search(query, k=10) == before


def test_round_trip_of_float64_vectors_keeps_the_answers_but_rounds_them():
    """The documented limit of storing float32: same neighbours in the same
    order, distances equal only to float32 precision."""
    index = build_index()
    query = np.random.default_rng(99).normal(size=8)
    before = index.search(query, k=10)

    restored = HNSWIndex.from_arrays(index.to_arrays(), dim=8, metric="euclidean")
    after = restored.search(query, k=10)

    assert [nid for nid, _ in after] == [nid for nid, _ in before]
    for (_, got), (_, want) in zip(after, before):
        assert got == pytest.approx(want, rel=1e-6)


def test_round_trip_preserves_the_graph_itself():
    index = build_index()
    restored = HNSWIndex.from_arrays(index.to_arrays(), dim=8, metric="euclidean")

    assert restored.entry_point == index.entry_point
    assert restored.max_layer == index.max_layer
    assert len(restored.layers) == len(index.layers)
    assert restored.layers == index.layers
    assert sorted(restored.vectors) == sorted(index.vectors)


def test_a_node_with_no_neighbours_survives_the_round_trip():
    """CSR exists to keep "present but isolated" distinct from "absent"."""
    index = HNSWIndex(dim=2, metric="euclidean", seed=1)
    index.insert(0, np.array([1.0, 0.0]))
    index.layers[0][99] = []
    index.vectors[99] = np.array([0.0, 1.0])

    restored = HNSWIndex.from_arrays(index.to_arrays(), dim=2, metric="euclidean")

    assert 99 in restored.layers[0]
    assert restored.layers[0][99] == []


def test_an_empty_index_round_trips():
    index = HNSWIndex(dim=4, metric="euclidean", seed=1)
    restored = HNSWIndex.from_arrays(index.to_arrays(), dim=4, metric="euclidean")

    assert len(restored) == 0
    assert restored.entry_point is None
    assert restored.max_layer == -1


def test_an_unknown_version_is_refused_rather_than_guessed_at():
    arrays = dict(build_index(n=5).to_arrays())
    arrays["hnsw_version"] = np.array(SNAPSHOT_VERSION + 1, dtype=np.int64)

    with pytest.raises(SnapshotError, match="version"):
        HNSWIndex.from_arrays(arrays, dim=8, metric="euclidean")


def test_a_missing_key_is_refused():
    arrays = dict(build_index(n=5).to_arrays())
    del arrays["hnsw_entry_point"]

    with pytest.raises(SnapshotError, match="missing"):
        HNSWIndex.from_arrays(arrays, dim=8, metric="euclidean")


def test_a_dim_mismatch_is_refused():
    arrays = build_index(n=5).to_arrays()

    with pytest.raises(SnapshotError, match="dim"):
        HNSWIndex.from_arrays(arrays, dim=16, metric="euclidean")


def test_an_edge_to_a_node_with_no_vector_is_refused():
    """Otherwise the dangling edge is followed during a later search and
    fails there, a long way from the snapshot that caused it."""
    index = build_index(n=10)
    arrays = dict(index.to_arrays())
    arrays["hnsw_L0_neighbors"] = np.append(
        arrays["hnsw_L0_neighbors"], np.array([9999], dtype=np.int64)
    )
    offsets = arrays["hnsw_L0_offsets"].copy()
    offsets[-1] += 1
    arrays["hnsw_L0_offsets"] = offsets

    with pytest.raises(SnapshotError, match="no vector"):
        HNSWIndex.from_arrays(arrays, dim=8, metric="euclidean")


def test_offsets_that_do_not_span_the_neighbours_are_refused():
    arrays = dict(build_index(n=10).to_arrays())
    offsets = arrays["hnsw_L0_offsets"].copy()
    offsets[-1] += 5
    arrays["hnsw_L0_offsets"] = offsets

    with pytest.raises(SnapshotError, match="offsets"):
        HNSWIndex.from_arrays(arrays, dim=8, metric="euclidean")


def test_garbage_is_refused_rather_than_raising_something_else():
    with pytest.raises(SnapshotError):
        HNSWIndex.from_arrays({"nonsense": np.array([1])}, dim=8)


def test_snapshot_error_is_catchable_as_value_error():
    assert issubclass(SnapshotError, ValueError)


# -- the VectorDB wrappers ---------------------------------------------------


def populated_db(n=40, dim=8):
    db = VectorDB(dim=dim, metric="euclidean", seed=1)
    rng = np.random.default_rng(3)
    for i in range(n):
        db.insert(f"id{i}", rng.normal(size=dim), metadata={"n": i})
    return db


def test_vectordb_round_trip_returns_identical_results_and_metadata():
    db = populated_db()
    query = np.random.default_rng(7).normal(size=8)
    before = db.search(query, k=5)

    restored = VectorDB(dim=8, metric="euclidean", seed=1)
    restored.restore_snapshot(
        db.snapshot_arrays(),
        live={f"id{i}": i for i in range(40)},
        metadata={f"id{i}": {"n": i} for i in range(40)},
        next_internal_id=40,
    )

    after = restored.search(query, k=5)
    assert [r["id"] for r in after] == [r["id"] for r in before]
    for got, want in zip(after, before):
        assert got["distance"] == pytest.approx(want["distance"], rel=1e-6)
    assert restored.get_metadata("id7") == {"n": 7}
    assert len(restored) == 40


def test_vectordb_round_trip_keeps_deleted_entries_out_of_results():
    db = populated_db()
    db.delete("id0")
    query = np.random.default_rng(7).normal(size=8)

    restored = VectorDB(dim=8, metric="euclidean", seed=1)
    restored.restore_snapshot(
        db.snapshot_arrays(),
        live={f"id{i}": i for i in range(1, 40)},
        deleted={"id0": 0},
        next_internal_id=40,
    )

    assert "id0" not in [r["id"] for r in restored.search(query, k=40)]
    assert len(restored) == 39


def test_a_restored_db_keeps_growing_the_same_way_a_fresh_one_does():
    """The seed has to survive the restore, or a restored db picks node
    levels from a different stream and stops being reproducible."""
    fresh = populated_db()
    rng = np.random.default_rng(11)
    extra = [rng.normal(size=8) for _ in range(5)]

    restored = VectorDB(dim=8, metric="euclidean", seed=1)
    restored.restore_snapshot(
        fresh.snapshot_arrays(),
        live={f"id{i}": i for i in range(40)},
        next_internal_id=40,
    )

    for i, vector in enumerate(extra):
        fresh.insert(f"new{i}", vector)
        restored.insert(f"new{i}", vector)

    assert restored.index.layers == fresh.index.layers
    assert restored.index.max_layer == fresh.index.max_layer


def test_a_bad_snapshot_leaves_the_db_untouched():
    db = populated_db(n=5)
    query = np.random.default_rng(7).normal(size=8)
    before = db.search(query, k=3)

    arrays = dict(db.snapshot_arrays())
    arrays["hnsw_version"] = np.array(SNAPSHOT_VERSION + 99, dtype=np.int64)

    with pytest.raises(SnapshotError):
        db.restore_snapshot(arrays, live={f"id{i}": i for i in range(5)})

    assert db.search(query, k=3) == before  # untouched, so still exact


def test_id_bookkeeping_that_disagrees_with_the_graph_is_still_refused():
    """restore_snapshot() has to keep restore_state()'s guarantees, not just
    from_arrays(): a graph node with no record is as wrong as a bad array."""
    db = populated_db(n=5)

    target = VectorDB(dim=8, metric="euclidean", seed=1)
    with pytest.raises(RestoreError):
        target.restore_snapshot(db.snapshot_arrays(), live={"id0": 0})
