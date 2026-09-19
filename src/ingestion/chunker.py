"""
Découpage du texte extrait en chunks pour l'indexation vectorielle.

TODO (semaine 1): remplacer ce découpage naïf par un chunking sémantique
(ex: RecursiveCharacterTextSplitter de LangChain) qui respecte les
paragraphes/sections plutôt que de couper au nombre de caractères.
"""

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    """Prend la sortie de pdf_parser.extract_text_by_page et retourne des chunks
    avec métadonnées (source du document + numéro de page) pour la traçabilité."""
    all_chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            if chunk.strip():
                all_chunks.append({"text": chunk, "source": source, "page": page["page"]})
    return all_chunks
