"""
Récupération des chunks pertinents depuis ChromaDB pour une question donnée.
"""

import functools
import os
import re
import unicodedata

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/processed/chroma")
COLLECTION_NAME = "financial_reports"


# Chargé paresseusement (voir la même remarque dans src/ingestion/indexer.py) :
# évite de déclencher un téléchargement réseau simplement en important ce module.
@functools.lru_cache(maxsize=1)
def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _normalize(text: str) -> str:
    """Minuscules, accents retirés, tout ce qui n'est pas alphanumérique
    remplacé par un espace (ex: "Délice Holding" et "delice_holding_2024.pdf"
    doivent se comparer sur un pied d'égalité malgré l'accent et le "_")."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


def _matches_source(source: str, source_filter: str) -> bool:
    """Vrai si `source_filter` désigne vraisemblablement `source`.

    Une simple sous-chaîne insensible à la casse ne suffit pas : le nom de
    fichier indexé ne reprend pas forcément l'intégralité de la raison
    sociale telle que l'utilisateur/le LLM la formule (ex: le PDF
    "exemple_rapport_test.pdf" correspond à la question "Société Exemple SA").
    On normalise donc (accents, séparateurs) puis on matche soit la chaîne
    complète collée, soit au moins un mot significatif (>= 4 caractères, pour
    ignorer les sigles comme "SA"/"SARL") du filtre dans la source.
    """
    norm_source = _normalize(source)
    norm_filter = _normalize(source_filter)
    if norm_filter.replace(" ", "") in norm_source.replace(" ", ""):
        return True
    tokens = [t for t in norm_filter.split(" ") if len(t) >= 4]
    return any(t in norm_source for t in tokens)


def retrieve(query: str, k: int = 5, source_filter: str | None = None) -> list[dict]:
    """Récupère les `k` chunks les plus pertinents pour `query`.

    Si `source_filter` est fourni, restreint les résultats aux chunks dont le
    champ `source` (nom du fichier PDF indexé) contient `source_filter`
    (comparaison insensible à la casse, sous-chaîne — le nom du fichier peut
    différer de la façon dont l'utilisateur nomme l'entreprise, ex.
    "delice_holding_2024.pdf" vs "Délice Holding"). Pour filtrer sans perdre
    en pertinence, on interroge ChromaDB sur nettement plus de résultats que
    `k` avant de filtrer en Python, puis on ne garde que les `k` premiers
    restants. Un simple facteur fixe (ex: k*4) s'est révélé insuffisant dès
    qu'un document est petit par rapport au reste du corpus (ex: 2 chunks
    face à 74 chunks d'un autre rapport) : ses chunks peuvent alors ne pas
    figurer parmi les k*4 meilleurs résultats bruts, faisant rater des
    incohérences réelles au sein même de ce document. On interroge donc la
    collection entière quand un filtre est actif (le corpus reste petit dans
    ce projet ; à revoir si le volume de documents indexés grossit beaucoup).
    Si aucun résultat ne matche après filtrage, retourne une liste vide
    plutôt que de retomber silencieusement sur les résultats non filtrés —
    un mélange de sources dans le contexte peut faire comparer les chiffres
    de deux entreprises différentes comme s'il s'agissait de la même (voir
    `src/rag/tools/inconsistency_detector.py`).
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=get_embedding_fn()
    )

    n_results = min(collection.count(), 500) if source_filter else k
    results = collection.query(query_texts=[query], n_results=n_results)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source"), "page": meta.get("page"), "score": dist})

    if source_filter:
        hits = [h for h in hits if h["source"] and _matches_source(h["source"], source_filter)]

    return hits[:k]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Quel est le résultat net ?"
    for hit in retrieve(query):
        print(f"[{hit['source']} p.{hit['page']}] (score={hit['score']:.3f})\n{hit['text'][:200]}\n")
