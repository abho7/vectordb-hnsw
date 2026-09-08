import pytest
import numpy as np

from vectordb.store import VectorDB


def test_insert_and_search_returns_metadata():
    db = VectorDB(dim=4, metric="euclidean", seed=1)
    db.insert("doc1", np.array([1.0, 0, 0, 0]), metadata={"title": "First"})
    results = db.search(np.array([1.0, 0, 0, 0]), k=1)
    assert results[0]["id"] == "doc1"
    assert results[0]["metadata"] == {"title": "First"}


def test_deleted_items_never_appear_in_search_results():
    db = VectorDB(dim=4, metric="euclidean", seed=1)
    db.insert("doc1", np.array([1.0, 0, 0, 0]))
    db.insert("doc2", np.array([0, 1.0, 0, 0]))
    db.insert("doc3", np.array([1.0, 0.1, 0, 0]))

    db.delete("doc1")
    results = db.search(np.array([1.0, 0, 0, 0]), k=3)
    ids = [r["id"] for r in results]
    assert "doc1" not in ids
    assert set(ids) == {"doc2", "doc3"}


def test_len_excludes_deleted():
    db = VectorDB(dim=4, seed=1)
    db.insert("a", np.zeros(4))
    db.insert("b", np.ones(4))
    assert len(db) == 2
    db.delete("a")
    assert len(db) == 1


def test_duplicate_insert_without_delete_raises():
    db = VectorDB(dim=4, seed=1)
    db.insert("a", np.zeros(4))
    try:
        db.insert("a", np.ones(4))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_reinsert_after_delete_succeeds():
    db = VectorDB(dim=4, seed=1)
    db.insert("a", np.zeros(4), metadata={"v": 1})
    db.delete("a")
    db.insert("a", np.ones(4), metadata={"v": 2})
    results = db.search(np.ones(4), k=1)
    assert results[0]["id"] == "a"
    assert results[0]["metadata"] == {"v": 2}


def test_delete_nonexistent_id_raises():
    db = VectorDB(dim=4, seed=1)
    try:
        db.delete("nonexistent")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_get_metadata_for_deleted_id_raises():
    db = VectorDB(dim=4, seed=1)
    db.insert("a", np.zeros(4), metadata={"x": 1})
    db.delete("a")
    try:
        db.get_metadata("a")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_reinsert_after_delete_removes_the_old_vector_not_just_hides_it():
    """A reinsert over a deleted id must retire the old vector for good.

    test_reinsert_after_delete_succeeds covers the same flow but queries
    with k=1 at the new vector's own location, so the new vector is
    always nearest and a surviving old one can never appear. This probes
    from the *old* vector's location with k > 1, which is where a
    resurrected node shows up.
    """
    db = VectorDB(dim=4, metric="euclidean", seed=1)
    old_position = np.array([1.0, 0, 0, 0])
    new_position = np.array([0, 1.0, 0, 0])

    db.insert("a", old_position, metadata={"v": 1})
    db.insert("filler1", np.array([0, 0, 1.0, 0]))
    db.insert("filler2", np.array([0, 0, 0, 1.0]))
    db.delete("a")
    db.insert("a", new_position, metadata={"v": 2})

    # Query from where the old vector used to be. If it is still in the
    # graph it is distance 0 from here and will come back first.
    # k covers every node in the index, so a resurrected duplicate has
    # room to appear alongside the new vector rather than being truncated.
    results = db.search(old_position, k=4)
    ids = [r["id"] for r in results]

    assert ids.count("a") == 1, f"'a' should appear once, got {ids}"

    # And the surviving entry has to be the *new* vector. Deduplicating
    # results by id would satisfy the count above while still returning
    # the stale vector, since from here the stale one is nearer.
    entry = next(r for r in results if r["id"] == "a")
    assert entry["distance"] == pytest.approx(np.sqrt(2.0)), (
        f"'a' should be at the new vector's distance, got {entry['distance']}"
    )
    assert entry["metadata"] == {"v": 2}


def test_len_is_correct_after_reinserting_a_deleted_id():
    db = VectorDB(dim=4, seed=1)
    db.insert("a", np.zeros(4))
    db.delete("a")
    db.insert("a", np.ones(4))
    assert len(db) == 1
