"""
Outil: récupération de contexte brut, pour les questions générales qui ne
correspondent à aucun outil spécialisé (ratios, incohérences, risques).

Reprend la même logique que get_relevant_context de src/mcp/server.py, avec
en plus le paramètre `company` (comme les 3 autres tools de src/rag/tools/)
pour restreindre la recherche au rapport d'une seule entreprise.
"""

from langchain_core.tools import tool

from src.rag.retriever import retrieve


@tool
def get_context(query: str, company: str = "") -> str:
    """Récupère des passages pertinents des rapports financiers indexés,
    pour répondre à une question générale qui ne correspond à aucun
    outil spécialisé (ratios, incohérences, risques). Si la question
    mentionne un nom d'entreprise précis, renseigne company pour
    restreindre la recherche à son rapport."""
    hits = retrieve(query, source_filter=company if company else None)
    if company and not hits:
        return f"Aucun rapport trouvé pour l'entreprise '{company}'."
    return "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)
