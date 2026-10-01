# Vector DB (HNSW from scratch)

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![tests](https://github.com/abho7/vectordb-hnsw/actions/workflows/tests.yml/badge.svg)](https://github.com/abho7/vectordb-hnsw/actions/workflows/tests.yml)

An approximate nearest-neighbor vector index implementing HNSW
(Hierarchical Navigable Small World graphs, Malkov & Yashunin 2016/2018)
from the primary paper -- the graph-based ANN algorithm underlying
Pinecone, Weaviate, and pgvector's HNSW index type. No FAISS, no
hnswlib, no wrapped C library: the multi-layer graph construction,
greedy layer descent, and bounded best-first search are implemented
directly in Python/NumPy.

**Verified, not assumed:** recall@10 = 1.000 against brute-force exact
search on clustered synthetic data at every tested scale (500 to 8,000
vectors), with query latency speedup over brute force growing from
1.8x to 20.1x as N increases -- the expected signature of O(log n) vs
O(n) scaling. That baseline is a pure-Python distance loop. Measured
instead against a vectorized NumPy scan, HNSW is about 3.3x faster at
8,000 vectors and slower at 500, where the graph traversal costs more
than scanning the whole set (see `results/scaling.jsonl`).
37/37 tests passing.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the algorithm, the layered
graph diagram, and two real correctness bugs found and fixed during
development -- including one where the graph's connectivity collapsed
to 4.8% of nodes on realistic clustered data despite passing every test
on uniform random data first.

## Why the paper's Algorithm 4 matters (and isn't optional)

The simplest version of HNSW's neighbor-selection rule -- keep the M
nearest candidates during graph construction -- looks correct on
uniform random test vectors. It silently produces a disconnected graph
on realistically clustered data (the kind real embeddings actually look
like), because a node deep in a tight cluster's nearest candidates are
almost always from that same cluster, so nothing ever forces the
long-range "bridge" edges the top layers depend on. This project's
benchmark caught it: recall stuck at 44% regardless of how large
`ef_search` was set, including `ef_search` equal to the entire dataset
-- proof the ceiling wasn't a tunable search parameter but an
unreachable-node problem. The fix is the paper's own proposed solution
(the diversity-aware SELECT-NEIGHBORS-HEURISTIC), and it's locked in as
a permanent regression test using clustered data specifically, since
uniform test data would never have caught the bug in the first place.

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v                        # 37 tests
python benchmarks/run_benchmark.py       # real recall@10 + speed vs. brute force
python scripts/demo_semantic_search.py    # TF-IDF + HNSW semantic search demo
```

## What's tested

| Property | File |
|---|---|
| Distance metric correctness (euclidean, cosine) | test_distance.py |
| Exact-match recall vs. brute force on small verifiable datasets | test_hnsw_correctness.py |
| Graph stays connected on clustered data (the core regression test) | test_hnsw_correctness.py |
| recall@10 > 0.9 on clustered data | test_hnsw_correctness.py |
| Metadata storage, soft-delete filtering | test_vectordb.py |
| TF-IDF fit/transform vocabulary consistency | test_tfidf.py |

## Project layout

```
src/
  hnsw/
    distance.py    # euclidean + cosine
    index.py         # HNSWIndex: Algorithm 1 (insert), 2 (search-layer), 4 (neighbor selection)
  vectordb/
    store.py          # string IDs, metadata, soft-delete
benchmarks/
  datasets.py       # Gaussian-cluster generator, from-scratch TF-IDF vectorizer
  run_benchmark.py    # recall@k + timing vs brute force
tests/                 # see table above
scripts/
  demo_semantic_search.py  # semantic search over disaster-response-style text
```

## Experiments

`experiments/` holds the scripts behind the measurements quoted above: the
reachability and recall collapse under the simple neighbor rule, a separation
sweep across cluster tightness, orphan-node counts, both rules on the UCI
handwritten digits, and the scaling run that produced the latency figures.
Raw output is in `results/`. Nothing in `src/` is modified to run them -- the
simple rule is a subclass that overrides one method.

See [experiments/README.md](./experiments/README.md) for what each script
measures and how to reproduce it.

## What's left to do

Stated in full in ARCHITECTURE.md's "What's simplified" section --
briefly: soft-delete only (no graph compaction), the diversity heuristic
without the paper's optional extendCandidates step, in-memory only (no
persistence), and no quantization for memory savings at very large
scale.
