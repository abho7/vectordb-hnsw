import pytest
import numpy as np

from hnsw.index import HNSWIndex
from vectordb.store import RestoreError, VectorDB


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


# ---------------------------------------------------------------- restore --
# restore_state() exists so callers persisting an index themselves do not have
# to assign to private attributes and model the internal representation
# correctly. These cover the contract it offers in exchange.


def _populated_db():
    db = VectorDB(dim=4, metric="euclidean", seed=1)
    db.insert("keep", np.array([1.0, 0, 0, 0]), metadata={"n": 1})
    db.insert("drop", np.array([0, 1.0, 0, 0]), metadata={"n": 2})
    db.delete("drop")
    return db


def test_restore_state_round_trips_a_live_and_a_deleted_entry():
    source = _populated_db()
    restored = VectorDB(dim=4, metric="euclidean", seed=1)
    restored.restore_state(
        source._index,
        live={"keep": 0},
        deleted={"drop": 1},
        metadata={"keep": {"n": 1}},
        next_internal_id=2,
    )

    assert len(restored) == 1
    results = restored.search(np.array([0, 1.0, 0, 0]), k=2)
    ids = [r["id"] for r in results]
    assert "drop" not in ids, "a restored tombstone must still filter that entry out"
    assert ids == ["keep"]
    assert restored.get_metadata("keep") == {"n": 1}


def test_restore_state_keeps_deleted_entries_from_crowding_out_live_ones():
    """The failure that motivated this method.

    A caller that assigned the wrong id currency to the private deleted set
    left tombstones inert: the deleted entry ranked normally, consumed the
    result budget, and the live entry fell off the end. Asking only whether
    the deleted id is absent does not catch that -- an empty result passes
    too -- so this asserts the live entry is present at k=1.
    """
    source = _populated_db()
    restored = VectorDB(dim=4, metric="euclidean", seed=1)
    restored.restore_state(
        source._index, live={"keep": 0}, deleted={"drop": 1}, metadata={"keep": {"n": 1}}
    )

    # Query sits on the deleted vector, so it wins on distance if still live.
    results = restored.search(np.array([0, 1.0, 0, 0]), k=1)
    assert [r["id"] for r in results] == ["keep"]


def test_restore_state_defaults_next_internal_id_past_the_highest_seen():
    source = _populated_db()
    restored = VectorDB(dim=4, metric="euclidean", seed=1)
    restored.restore_state(source._index, live={"keep": 0}, deleted={"drop": 1})

    # A fresh insert must not collide with a restored id.
    restored.insert("third", np.array([0, 0, 1.0, 0]))
    assert restored._id_map["third"] == 2
    assert len(restored) == 2


def test_restore_state_rejects_incoherent_state_and_leaves_the_db_untouched():
    source = _populated_db()
    db = VectorDB(dim=4, metric="euclidean", seed=1)
    db.insert("original", np.array([1.0, 0, 0, 0]), metadata={"kept": True})
    before = len(db)

    cases = {
        "node with no external id": dict(live={"keep": 0}),
        "id both live and deleted": dict(live={"keep": 0, "drop": 1}, deleted={"drop": 1}),
        "duplicate internal ids": dict(live={"keep": 0}, deleted={"drop": 0}),
        "metadata for a deleted id": dict(
            live={"keep": 0}, deleted={"drop": 1}, metadata={"drop": {}}
        ),
        "next_internal_id reuses an id": dict(
            live={"keep": 0}, deleted={"drop": 1}, next_internal_id=1
        ),
    }
    for label, kwargs in cases.items():
        with pytest.raises(RestoreError):
            db.restore_state(source._index, **kwargs)

    assert len(db) == before, "a rejected restore must not modify the db"
    assert db.get_metadata("original") == {"kept": True}
    assert [r["id"] for r in db.search(np.array([1.0, 0, 0, 0]), k=1)] == ["original"]


def test_restore_state_rejects_a_mismatched_index():
    db = VectorDB(dim=4, metric="euclidean", seed=1)
    with pytest.raises(RestoreError):
        db.restore_state(HNSWIndex(dim=8, metric="euclidean"), live={})
    with pytest.raises(RestoreError):
        db.restore_state(HNSWIndex(dim=4, metric="cosine"), live={})


def test_restore_state_accepts_an_empty_index():
    db = VectorDB(dim=4, metric="euclidean", seed=1)
    db.restore_state(HNSWIndex(dim=4, metric="euclidean"), live={})
    assert len(db) == 0
    assert db.search(np.zeros(4), k=1) == []


def test_index_property_exposes_the_graph_for_serialisation():
    db = _populated_db()
    assert db.index is db._index
    # The pairing that matters: read the graph out, hand it back in.
    restored = VectorDB(dim=4, metric="euclidean", seed=1)
    restored.restore_state(db.index, live={"keep": 0}, deleted={"drop": 1})
    assert [r["id"] for r in restored.search(np.array([1.0, 0, 0, 0]), k=2)] == ["keep"]
