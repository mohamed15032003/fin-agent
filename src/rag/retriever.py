"""
Récupération des chunks pertinents depuis ChromaDB pour une question donnée.
"""

import functools
import os

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/processed/chroma")
COLLECTION_NAME = "financial_reports"


# Chargé paresseusement (voir la même remarque dans src/ingestion/indexer.py) :
# évite de déclencher un téléchargement réseau simplement en important ce module.
@functools.lru_cache(maxsize=1)
def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def retrieve(query: str, k: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=get_embedding_fn()
    )

    results = collection.query(query_texts=[query], n_results=k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source"), "page": meta.get("page"), "score": dist})
    return hits


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Quel est le résultat net ?"
    for hit in retrieve(query):
        print(f"[{hit['source']} p.{hit['page']}] (score={hit['score']:.3f})\n{hit['text'][:200]}\n")
