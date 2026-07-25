import numpy as np

from datasets import TfidfVectorizer, tfidf_embeddings


def test_fit_transform_matches_direct_convenience_function():
    docs = ["the cat sat on the mat", "the dog ran in the park"]
    vectors, vocab = tfidf_embeddings(docs)
    vectorizer = TfidfVectorizer().fit(docs)
    for i, doc in enumerate(docs):
        assert np.allclose(vectors[i], vectorizer.transform(doc))
    assert vocab == vectorizer.vocab


def test_query_with_unseen_word_keeps_fixed_dimension():
    """Regression test for a real bug found while building the semantic
    search demo: re-fitting a vectorizer on (corpus + query) changes the
    vocabulary size the moment the query has any new word, producing an
    embedding whose dimension doesn't match the already-indexed
    documents -- which crashes vector search. transform() on an
    already-fitted vectorizer must never change the output dimension,
    regardless of what words appear in the input text."""
    docs = ["wildfire evacuation shelter", "stock market rally today"]
    vectorizer = TfidfVectorizer().fit(docs)
    dim_before = len(vectorizer.vocab)

    query_with_new_word = "wildfire evacuation antidisestablishmentarianism"
    query_vec = vectorizer.transform(query_with_new_word)

    assert query_vec.shape == (dim_before,)
    assert len(vectorizer.vocab) == dim_before


def test_unseen_words_are_silently_dropped_not_erroring():
    vectorizer = TfidfVectorizer().fit(["hello world"])
    vec = vectorizer.transform("completely different vocabulary entirely")
    assert vec.shape == (len(vectorizer.vocab),)


def test_empty_text_returns_zero_vector():
    vectorizer = TfidfVectorizer().fit(["hello world", "foo bar"])
    vec = vectorizer.transform("")
    assert np.allclose(vec, np.zeros(len(vectorizer.vocab)))


def test_identical_documents_have_identical_embeddings():
    docs = ["wildfire near the city", "wildfire near the city", "completely unrelated text"]
    vectorizer = TfidfVectorizer().fit(docs)
    v0 = vectorizer.transform(docs[0])
    v1 = vectorizer.transform(docs[1])
    assert np.allclose(v0, v1)
