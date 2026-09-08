# Architecture

## 1. The algorithm

HNSW (Malkov & Yashunin, "Efficient and Robust Approximate Nearest
Neighbor Search Using Hierarchical Navigable Small World Graphs,"
2016/2018) builds a multi-layer proximity graph:

```
Layer 2:        A -------------------- F                (sparse: long-range "highway" edges)
                 |                      |
Layer 1:        A ---- C -------- D --- F ---- H          (medium density)
                 |      |          |    |       |
Layer 0:  A - B - C - D - E - F - G - H - I - J - K - ...  (dense: every point's true near neighbors)
```

A query starts at a random entry point in the top layer and greedily
descends: at each layer, walk to the closest neighbor, drop down a
layer, repeat. By the time the query reaches layer 0, it's already in
roughly the right neighborhood, and a bounded best-first search
(`ef_search` candidates) at layer 0 finds the actual answer. This is
what gives HNSW expected O(log n) search instead of brute force's O(n)
-- the top layers act like an index into an index.

## 2. A real bug found by testing at realistic scale, not toy examples

**Building the graph with SELECT-NEIGHBORS-SIMPLE (the paper's basic
neighbor-selection rule: just keep the M nearest candidates) collapsed
the entire index.** On uniformly random test vectors, this rule looks
fine -- 98.6% of nodes stayed reachable from the entry point, and
early correctness tests (small N, verified against brute force) passed
cleanly. It was only when benchmarking on realistically clustered data
-- the kind actual embeddings look like, since real documents and
images cluster into semantic groups rather than spreading uniformly
through the vector space -- that the graph fell apart: only 96 of 2,000
nodes (4.8%) were reachable from the entry point. Recall stayed locked
at exactly 49.5% no matter how large `ef_search` was set, including
`ef_search` equal to the entire dataset size, which is what proved it
wasn't a tunable search-quality problem -- unreachable nodes can't be
found by any search budget, however large.

**Root cause:** for a node deep inside a tight cluster, its M nearest
candidates during insertion are almost always all from that same
cluster. Nothing in the "just take the M closest" rule ever forces a
long-range "bridge" edge to a different part of the vector space. The
top layers, which are supposed to provide exactly those bridges, never
get any -- so the graph fragments into per-cluster islands with no path
between them.

**Fix:** the paper's own proposed solution, SELECT-NEIGHBORS-HEURISTIC
(Algorithm 4) -- a diversity-aware selection rule. A candidate is only
kept if it's closer to the query than it is to every neighbor already
selected; candidates that are redundant with an already-picked neighbor
get rejected even if they're individually close, which is exactly what
forces some connections to point outward across cluster boundaries
instead of only inward. After switching to this heuristic:

| Metric | SELECT-NEIGHBORS-SIMPLE | SELECT-NEIGHBORS-HEURISTIC |
|---|---|---|
| Nodes reachable from entry point (clustered, n=2000) | 96 / 2000 (4.8%) | 2000 / 2000 (100%) |
| recall@10 (clustered data) | stuck at 0.495 regardless of ef_search | 1.000 |

This is locked in as a permanent regression test
(`tests/test_hnsw_correctness.py::test_graph_remains_connected_on_clustered_data`)
specifically using clustered, not uniform, synthetic data -- uniform
data would never have caught this, which is itself the lesson: an ANN
index needs to be benchmarked against data that actually resembles what
it'll index in production, not the easiest case that happens to work.

## 3. A second bug: fit/transform vocabulary drift

While building the semantic-search demo, re-embedding a query by
re-running TF-IDF over (corpus + query) silently changed the vocabulary
size the moment the query contained any word not already in the corpus
-- producing a query vector with a different dimension than the
already-indexed documents, which breaks vector search outright. The fix
mirrors why scikit-learn separates `fit()` from `transform()`:
`benchmarks/datasets.py::TfidfVectorizer` fits a vocabulary once from
the corpus and reuses that fixed vocabulary (silently dropping
out-of-vocabulary words, the standard and correct behavior) for every
later `transform()` call, including queries. Regression test:
`tests/test_tfidf.py::test_query_with_unseen_word_keeps_fixed_dimension`.

## 4. Real benchmark numbers (recall@10, euclidean, dim=32, Gaussian clusters)

Run with `python benchmarks/run_benchmark.py`:

| N | build (s) | recall@10 | HNSW query (ms) | brute-force query (ms) | speedup |
|---|---|---|---|---|---|
| 500 | 5.1 | 1.000 | 0.78 | 1.40 | 1.8x |
| 2,000 | 23.3 | 1.000 | 0.89 | 5.41 | 6.1x |
| 8,000 | 105.0 | 1.000 | 1.35 | 27.17 | 20.1x |

The speedup growing with N (not staying constant) is the expected
signature of O(log n) vs O(n) scaling -- brute force's cost grows
linearly with the dataset, HNSW's grows much more slowly, so the gap
widens as N increases. Recall stays at 1.000 across all three sizes
after the Algorithm 4 fix.

## 5. Project layout

```
src/
  hnsw/
    distance.py     # euclidean + cosine, single and batched
    index.py          # HNSWIndex: the actual algorithm, Algorithm 1/2/4 from the paper
  vectordb/
    store.py           # user-facing API: string IDs, metadata, soft-delete
tests/
  test_distance.py       # distance function correctness
  test_hnsw_correctness.py  # exact-match recall, the clustered-data connectivity regression test
  test_vectordb.py          # metadata, soft-delete filtering
  test_tfidf.py               # the fit/transform vocabulary-drift regression test
benchmarks/
  datasets.py         # Gaussian-cluster generator, from-scratch TfidfVectorizer
  run_benchmark.py      # recall@k + timing vs brute force
scripts/
  demo_semantic_search.py  # TF-IDF + HNSW over disaster-response-style text
```

## 6. What's simplified, stated plainly

- **Soft-delete only, no compaction.** Deleted entries are filtered from
  results but never removed from the graph structure, so the index only
  grows. A real system needs a background compaction/rebuild pass;
  documented as future work, not implemented. Tombstones are keyed by
  *internal* node id, not by the caller's external id, which is what makes
  reusing an id safe: `insert()` after `delete()` allocates a fresh node and
  leaves the previous one tombstoned for good, so the superseded vector can
  never resurface. Keying them by external id instead resurrected the old
  node on reuse, leaving two live nodes under one id
  (`tests/test_vectordb.py::test_reinsert_after_delete_removes_the_old_vector_not_just_hides_it`).
  Callers that persist an index themselves bring it back through
  `VectorDB.restore_state()`, which takes external ids plus which of them are
  deleted and does its own translation. It exists because the internal keying
  above is exactly the kind of detail a caller should not have to model: when
  it changed, a downstream project that had been assigning to the private
  attributes directly kept running and silently stopped filtering deleted
  entries. `restore_state()` validates that the graph, the id maps and the
  tombstones describe a coherent index, and raises `RestoreError` without
  touching the instance if they do not, so the caller can fall back to a
  rebuild.
- **SELECT-NEIGHBORS-HEURISTIC without the extendCandidates option**
  from the paper (which would also explore each candidate's own
  neighbors before selecting, for even better graph quality at higher
  construction cost). The version implemented here is the diversity
  rule alone, which was sufficient to fix the connectivity collapse and
  reach recall@10 = 1.000 on the tested cluster configurations.
- **In-memory only**, no persistence to disk and no incremental
  index-loading. Every process restart rebuilds from scratch.
- **No approximate distance computation** (e.g. product quantization or
  scalar quantization for memory savings at very large N) -- every
  vector is stored and compared at full precision.
