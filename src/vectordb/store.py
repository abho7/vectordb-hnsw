"""
User-facing VectorDB API: wraps HNSWIndex with string IDs, arbitrary
metadata storage, and soft-delete.

Soft-delete, not hard-delete: actually removing a node from an HNSW
graph is nontrivial -- its neighbors' edge lists would need repair to
avoid leaving dangling references, and doing that naively can damage the
graph's connectivity guarantees (the same failure mode this project's
own testing already found once, from a different cause). Real systems
(Pinecone, Weaviate) handle this the same way: mark deleted, filter
matching entries out of search results, and reclaim the space later via
a background compaction/rebuild pass rather than an in-place graph edit.
That compaction pass is listed as future work rather than implemented
here -- see ARCHITECTURE.md.
"""

from __future__ import annotations

import numpy as np

from hnsw.index import HNSWIndex


class VectorDB:
    def __init__(self, dim: int, *, metric: str = "cosine", M: int = 16, ef_construction: int = 200, seed: int | None = None):
        self._index = HNSWIndex(dim=dim, metric=metric, M=M, ef_construction=ef_construction, seed=seed)
        self._id_map: dict[str, int] = {}  # external string id -> internal int id
        self._reverse_id_map: dict[int, str] = {}
        self._metadata: dict[str, dict] = {}
        # Tombstones are keyed by *internal* id, because that is what the
        # graph holds. Keying them by external id meant that reusing an
        # external id (delete then insert) lifted the tombstone covering the
        # previous vector's node, leaving it live in the graph under the same
        # id as the new one. Internal ids are never reused, so a tombstone
        # here retires exactly one node, permanently.
        self._deleted: set[int] = set()
        self._next_internal_id = 0

    def __len__(self) -> int:
        return sum(1 for internal_id in self._id_map.values() if internal_id not in self._deleted)

    def insert(self, external_id: str, vector: np.ndarray, metadata: dict | None = None) -> None:
        previous_internal_id = self._id_map.get(external_id)
        if previous_internal_id is not None and previous_internal_id not in self._deleted:
            raise ValueError(f"id {external_id!r} already exists (use delete() first to replace it)")

        # Reinserting over a deleted id: the previous node stays in the graph
        # (soft delete), so its tombstone has to stay with it. Nothing here
        # clears it -- the mapping below just stops pointing at it.
        internal_id = self._next_internal_id
        self._next_internal_id += 1
        self._index.insert(internal_id, vector)
        self._id_map[external_id] = internal_id
        self._reverse_id_map[internal_id] = external_id
        self._metadata[external_id] = metadata or {}

    def delete(self, external_id: str) -> None:
        if external_id not in self._id_map:
            raise KeyError(f"id {external_id!r} not found")
        self._deleted.add(self._id_map[external_id])

    def search(self, query: np.ndarray, k: int, ef_search: int | None = None) -> list[dict]:
        """Returns up to k results, each {"id": ..., "distance": ...,
        "metadata": ...}, sorted nearest first. Over-fetches internally
        to compensate for filtering out soft-deleted entries, since a
        naive top-k from the raw index could return fewer than k live
        results if some of the true nearest neighbors were deleted."""
        fetch_k = k
        results: list[dict] = []
        seen_internal_ids: set[int] = set()

        # Retry with a larger fetch if deletions ate into the result count,
        # bounded so a pathological all-deleted index doesn't loop forever.
        for _ in range(5):
            raw = self._index.search(query, k=fetch_k, ef_search=ef_search)
            results = []
            for internal_id, dist in raw:
                if internal_id in self._deleted:
                    continue
                external_id = self._reverse_id_map[internal_id]
                results.append({"id": external_id, "distance": dist, "metadata": self._metadata[external_id]})
            if len(results) >= k or fetch_k >= len(self._index):
                break
            fetch_k *= 4

        return results[:k]

    def get_metadata(self, external_id: str) -> dict:
        if external_id not in self._id_map or self._id_map[external_id] in self._deleted:
            raise KeyError(f"id {external_id!r} not found")
        return self._metadata[external_id]
