"""
Interface Streamlit pour FinAgent : analyse de rapports financiers via
RAG (ChromaDB) + agent LangChain avec tool-calling (ratios, incohérences,
résumé des risques).

Usage local :
    streamlit run streamlit_app.py
(à lancer depuis la racine du projet, comme les autres entrypoints —
`uvicorn src.api.main:app`, `python -m src.rag.agent` — pour que les
imports `src.*` et les chemins relatifs de données se résolvent correctement.)
"""

import os
import re

import chromadb
import streamlit as st
from dotenv import load_dotenv

from src.ingestion.indexer import build_index
from src.rag.agent import answer
from src.rag.retriever import COLLECTION_NAME, PERSIST_DIR, get_embedding_fn

load_dotenv()

st.set_page_config(page_title="FinAgent", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner=False)
def _get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=get_embedding_fn())


def _ensure_indexed() -> None:
    """Reconstruit l'index ChromaDB si la collection est vide — cas d'un
    premier démarrage après un clone frais du repo, où data/processed/
    (l'index) n'existe pas encore puisqu'il est volontairement exclu du
    dépôt (régénérable, contrairement aux PDF sources dans data/raw/ qui,
    eux, sont commités). Réutilise build_index() telle quelle (la même
    fonction que `python -m src.ingestion.indexer` appelle en CLI), sans
    dupliquer sa logique."""
    if _get_collection().count() == 0:
        with st.spinner("Première initialisation : indexation des rapports financiers..."):
            build_index()
        _get_collection.clear()  # le client/collection mis en cache doit être recréé après l'indexation


def get_indexed_sources() -> list[str]:
    """Liste les documents (PDF) actuellement indexés dans ChromaDB, en
    lisant directement les métadonnées de la collection — jamais codé en
    dur, pour rester à jour à mesure que de nouveaux rapports sont indexés."""
    try:
        metadatas = _get_collection().get(include=["metadatas"])["metadatas"]
    except Exception:
        return []
    return sorted({m["source"] for m in metadatas if m.get("source")})


def friendly_name(filename: str) -> str:
    """Nom de fichier -> libellé lisible pour les exemples de questions
    (ex: "delice_holding_2024.pdf" -> "Delice Holding 2024")."""
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"[_-]+", " ", name).strip()
    return name.title()


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("FinAgent analyse les rapports financiers indexés..."):
        response = answer(question)
    st.session_state.messages.append({"role": "assistant", "content": response})


if "messages" not in st.session_state:
    st.session_state.messages = []

_ensure_indexed()

st.title("📊 FinAgent")
st.caption(
    "Analyse de rapports financiers via RAG (ChromaDB) + agent LangChain avec "
    "tool-calling (ratios financiers, détection d'incohérences, résumé des "
    "facteurs de risque). Les réponses sont générées uniquement à partir des "
    "rapports indexés ci-contre."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY n'est pas configurée. En local, renseigne-la dans `.env` "
        "(voir `.env.example`) ; sur Streamlit Community Cloud, ajoute-la dans "
        "les *Secrets* de l'application (voir `.streamlit/secrets.toml.example`)."
    )

sources = get_indexed_sources()

with st.sidebar:
    st.header("Rapports indexés")
    if sources:
        for s in sources:
            st.markdown(f"- **{friendly_name(s)}**  \n  `{s}`")
    else:
        st.warning(
            "Aucun document indexé pour le moment. Lance "
            "`python -m src.ingestion.indexer` après avoir déposé des PDF "
            "dans `data/raw/`."
        )

    st.divider()
    st.header("Exemples de questions")
    if sources:
        example_company = st.selectbox(
            "Entreprise pour les exemples", [friendly_name(s) for s in sources]
        )
        if st.button("📈 Ratios financiers", use_container_width=True):
            ask(f"Quels sont les ratios financiers de {example_company} ?")
        if st.button("⚠️ Incohérences", use_container_width=True):
            ask(f"Y a-t-il des incohérences dans les états financiers de {example_company} ?")
        if st.button("🛡️ Facteurs de risque", use_container_width=True):
            ask(f"Quels sont les principaux facteurs de risque de {example_company} ?")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Pose une question sur les rapports financiers indexés..."):
    ask(question)
    st.rerun()
