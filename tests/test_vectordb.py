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
