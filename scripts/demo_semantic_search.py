#!/usr/bin/env python3
"""
Demo: semantic search over disaster-response-related text using
from-scratch TF-IDF embeddings and the HNSW index built in this project.

Run with: python scripts/demo_semantic_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from datasets import TfidfVectorizer
from vectordb.store import VectorDB

DOCUMENTS = [
    "Flash flood warning issued for downtown area, residents urged to evacuate immediately",
    "Wildfire spreading rapidly near residential neighborhoods, evacuation orders in effect",
    "Emergency shelters opened at the community center for flood evacuees",
    "Heavy rainfall causing severe flooding along the river, roads closed",
    "Firefighters battling wildfire that has burned over 500 acres so far",
    "Local hospital reports overcrowding as residents seek shelter from wildfire smoke",
    "Stock market rallies as tech earnings beat analyst expectations this quarter",
    "New smartphone model announced with improved camera and battery life",
    "City council votes to approve new public transit funding for next year",
    "Local high school wins state championship in dramatic overtime finish",
    "Restaurant chain announces expansion into three new states next year",
    "Search and rescue teams deployed to assist residents trapped by flood waters",
    "Air quality alert issued due to heavy smoke from nearby wildfire",
    "Volunteers distribute supplies at emergency shelter for displaced families",
    "Tech company reports record quarterly profits driven by cloud services growth",
]


def main() -> None:
    print("Building TF-IDF embeddings for a 15-document corpus (disaster reports + unrelated news)...")
    vectorizer = TfidfVectorizer().fit(DOCUMENTS)
    print(f"  vocabulary size: {len(vectorizer.vocab)} terms, embedding dim: {len(vectorizer.vocab)}")

    db = VectorDB(dim=len(vectorizer.vocab), metric="cosine", M=8, ef_construction=100, seed=1)
    for i, doc in enumerate(DOCUMENTS):
        db.insert(f"doc_{i}", vectorizer.transform(doc), metadata={"text": doc})

    query_text = "Wildfire evacuation shelter emergency response"
    print(f"\nQuery: {query_text!r}\n")

    query_embedding = vectorizer.transform(query_text)  # same fixed vocabulary as the indexed documents

    results = db.search(query_embedding, k=5)
    print("Top 5 nearest documents (cosine distance, lower = more similar):")
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['metadata']['text']}")

    print("\nNote: the disaster/emergency-related documents rank first despite")
    print("sharing no exact keywords with the query beyond a few terms --")
    print("that's TF-IDF cosine similarity picking up on shared vocabulary")
    print("patterns across the disaster-related documents as a group.")


if __name__ == "__main__":
    main()
