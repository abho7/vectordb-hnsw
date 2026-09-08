"""
Dataset generators for benchmarking. Two kinds:

1. Synthetic Gaussian-cluster vectors -- the standard way ANN algorithms
   are benchmarked in the literature when real embedding downloads
   aren't available; clustered synthetic data still exercises the same
   distance-concentration behavior that makes high-dimensional ANN
   search hard.

2. A from-scratch TF-IDF text embedding generator -- no pretrained
   model or network access required, but real enough to demonstrate the
   actual downstream use case (searching over document embeddings)
   instead of only abstract random vectors.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def gaussian_clusters(n: int, dim: int, n_clusters: int = 20, cluster_std: float = 0.3, seed: int = 0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=3.0, size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n)
    vectors = {}
    for i in range(n):
        vectors[i] = centers[assignments[i]] + rng.normal(scale=cluster_std, size=dim)
    return vectors


_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "and", "or", "for", "with"}


def _tokenize(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w not in _STOPWORDS]


class TfidfVectorizer:
    """Deliberately mirrors sklearn's fit/transform split, for the same
    reason sklearn has it: a query embedded against a freshly-recomputed
    vocabulary (rather than the corpus's fixed vocabulary) can end up
    with a different dimensionality than the indexed documents the
    moment the query contains any word not already in the corpus --
    which silently breaks vector search the instant it happens. This
    class makes that mistake structurally harder to make by keeping the
    fitted vocabulary as state, separate from encoding any individual
    piece of text."""

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self.vocab_index: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._n_docs = 0

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        tokenized = [_tokenize(doc) for doc in documents]
        self.vocab = sorted({w for tokens in tokenized for w in tokens})
        self.vocab_index = {w: i for i, w in enumerate(self.vocab)}
        self._n_docs = len(documents)

        doc_freq = Counter()
        for tokens in tokenized:
            for w in set(tokens):
                doc_freq[w] += 1
        self.idf = {w: math.log(self._n_docs / (1 + doc_freq[w])) + 1 for w in self.vocab}
        return self

    def transform(self, text: str) -> np.ndarray:
        """Embed a single piece of text against the FIXED, already-fitted
        vocabulary. Words not seen during fit() are silently dropped --
        the standard, correct behavior for out-of-vocabulary query terms,
        not a bug: there's no way to place an unseen word in a
        vocabulary-indexed space."""
        vec = np.zeros(len(self.vocab))
        tokens = _tokenize(text)
        if not tokens:
            return vec
        term_counts = Counter(tokens)
        for w, count in term_counts.items():
            if w not in self.vocab_index:
                continue
            tf = count / len(tokens)
            vec[self.vocab_index[w]] = tf * self.idf[w]
        return vec

    def fit_transform(self, documents: list[str]) -> dict[int, np.ndarray]:
        self.fit(documents)
        return {i: self.transform(doc) for i, doc in enumerate(documents)}


def tfidf_embeddings(documents: list[str]) -> tuple[dict[int, np.ndarray], list[str]]:
    """Convenience wrapper for the common case (embed a fixed corpus,
    no separate query). For anything that also needs to embed queries
    against the same vocabulary later, use TfidfVectorizer directly --
    see scripts/demo_semantic_search.py."""
    vectorizer = TfidfVectorizer().fit(documents)
    vectors = {i: vectorizer.transform(doc) for i, doc in enumerate(documents)}
    return vectors, vectorizer.vocab
