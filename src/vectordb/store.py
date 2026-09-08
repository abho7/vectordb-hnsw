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

from collections.abc import Mapping

import numpy as np

from hnsw.index import HNSWIndex


class RestoreError(ValueError):
    """The state handed to VectorDB.restore_state() is not self-consistent.

    A ValueError subclass so that callers already catching ValueError around
    a restore keep working, while callers that want to fall back to a rebuild
    can catch this specifically.
    """


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

    @property
    def index(self) -> HNSWIndex:
        """The underlying graph, for callers that persist it themselves.

        The counterpart to restore_state(): read the graph out to serialise
        it, hand it back in to adopt it. Treat it as read-only -- mutating it
        behind this class's back leaves the id bookkeeping describing a graph
        that no longer exists.
        """
        return self._index

    def restore_state(
        self,
        index: HNSWIndex,
        *,
        live: Mapping[str, int],
        deleted: Mapping[str, int] = {},
        metadata: Mapping[str, dict] | None = None,
        next_internal_id: int | None = None,
    ) -> None:
        """Adopt a graph and id bookkeeping loaded from somewhere else.

        For callers that persist an index themselves and want to bring it
        back without paying to rebuild it. Everything is expressed in the
        caller's own currency -- external ids, and which of them are deleted
        -- so that how this class tracks deletions internally stays its own
        business. That is the point of the method: a caller assigning to the
        private attributes directly has to model the internal representation
        correctly, and when that representation changed, callers silently
        broke.

        live/deleted map external id -> internal id, the same internal ids the
        nodes carry in `index`. metadata is keyed by external id and applies
        to live entries. next_internal_id defaults to one past the highest
        internal id seen, which is the only safe value if it is not tracked
        separately.

        Raises RestoreError if the pieces do not describe a coherent index,
        leaving this instance untouched, so a caller can fall back to
        rebuilding from source data.
        """
        if index.dim != self._index.dim:
            raise RestoreError(f"index dim {index.dim} != this db's dim {self._index.dim}")
        if index.metric != self._index.metric:
            raise RestoreError(f"index metric {index.metric!r} != this db's {self._index.metric!r}")

        overlap = live.keys() & deleted.keys()
        if overlap:
            raise RestoreError(f"ids are both live and deleted: {sorted(overlap)[:5]}")

        combined: dict[str, int] = {**live, **deleted}
        if len(set(combined.values())) != len(combined):
            raise RestoreError("internal ids are not unique across live and deleted entries")

        # Every node the graph can walk to has to resolve to an external id,
        # or a search that reaches it raises partway through.
        unmapped = set(index.vectors) - set(combined.values())
        if unmapped:
            raise RestoreError(
                f"{len(unmapped)} node(s) in the graph have no external id, "
                f"e.g. {sorted(unmapped)[:5]}"
            )

        metadata = metadata or {}
        unknown = metadata.keys() - live.keys()
        if unknown:
            raise RestoreError(f"metadata for ids that are not live: {sorted(unknown)[:5]}")

        highest = max(combined.values(), default=-1)
        if next_internal_id is None:
            next_internal_id = highest + 1
        elif next_internal_id <= highest:
            raise RestoreError(
                f"next_internal_id {next_internal_id} would reuse an internal id "
                f"already in use (highest is {highest})"
            )

        # Nothing above this line mutates self, so a rejected restore leaves
        # the instance exactly as it was.
        self._index = index
        self._id_map = dict(combined)
        self._reverse_id_map = {internal: external for external, internal in combined.items()}
        self._metadata = {external_id: dict(metadata.get(external_id, {})) for external_id in live}
        self._deleted = set(deleted.values())
        self._next_internal_id = next_internal_id
