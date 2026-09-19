"""
Pipeline d'ingestion complet : PDF -> texte -> chunks -> embeddings -> ChromaDB.

Usage:
    python -m src.ingestion.indexer
"""

import functools
import glob
import os

import chromadb
from chromadb.utils import embedding_functions

from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_parser import extract_text_by_page

RAW_DIR = os.path.join("data", "raw")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/processed/chroma")
COLLECTION_NAME = "financial_reports"


# TODO (semaine 1): évaluer un modèle d'embeddings mieux adapté au français/finance
# si les rapports BVMT sont en français (ex: modèle multilingue sentence-transformers).
#
# Chargé paresseusement (lazy) via cette fonction plutôt qu'au niveau du module :
# le premier appel télécharge le modèle depuis huggingface.co (connexion internet
# requise, une seule fois — le modèle est ensuite mis en cache localement). Le
# charger au niveau module forcerait cet appel réseau dès qu'on importe ce
# fichier, y compris pour des tests qui n'en ont pas besoin.
@functools.lru_cache(maxsize=1)
def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def build_index() -> None:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=get_embedding_fn()
    )

    pdf_paths = glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"Aucun PDF trouvé dans {RAW_DIR}. Lance d'abord download_samples.py.")
        return

    doc_id = 0
    for pdf_path in pdf_paths:
        print(f"Traitement: {pdf_path}")
        pages = extract_text_by_page(pdf_path)
        chunks = chunk_pages(pages, source=os.path.basename(pdf_path))

        ids = [f"doc_{doc_id + i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        doc_id += len(chunks)
        print(f"  -> {len(chunks)} chunks indexés")

    print(f"\nIndex construit dans {PERSIST_DIR} ({doc_id} chunks au total).")


if __name__ == "__main__":
    build_index()
