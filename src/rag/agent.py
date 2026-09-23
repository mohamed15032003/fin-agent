"""
Agent orchestrateur : reçoit une question et la transmet à un agent LangChain
(create_agent) qui décide, via le tool-calling natif du modèle Groq, quel
outil appeler (ratios, incohérences, résumé de risques, ou recherche de
contexte générale) avant de produire une réponse.

Aucun contexte n'est pré-injecté ici : c'est aux tools d'aller chercher
eux-mêmes le contexte pertinent (chacun fait son propre retrieve()). Une
pré-injection systématique d'un contexte non filtré empêchait l'agent
d'utiliser le paramètre `company` des tools quand un document plus petit
était noyé par un autre plus volumineux dans les résultats bruts.
"""

import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from src.rag.tools.ratios import compute_ratios
from src.rag.tools.inconsistency_detector import detect_inconsistencies
from src.rag.tools.risk_summarizer import summarize_risks
from src.rag.tools.context import get_context

# llama-3.3-70b-versatile a ete retire par Groq le 16 aout 2026 ; remplacement
# officiellement recommande par Groq (voir console.groq.com/docs/deprecations).
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """Tu es un assistant d'analyse financière. Réponds uniquement
à partir des rapports financiers indexés, jamais de connaissances générales.
Si l'information n'est pas trouvée par les outils, dis-le clairement plutôt
que d'inventer.

Tu n'as accès à aucun contexte au départ : tu dois systématiquement appeler
un outil pour aller chercher l'information dont tu as besoin. Utilise les
outils spécialisés (ratios financiers, incohérences chiffrées, résumé des
facteurs de risque) quand la question s'y prête ; sinon utilise l'outil
général de recherche de contexte. Si la question mentionne un nom
d'entreprise ou de société précis, renseigne le paramètre company de
l'outil choisi pour restreindre la recherche au rapport de cette seule
entreprise."""

_model = ChatGroq(model=MODEL, api_key=os.environ.get("GROQ_API_KEY"))

_agent = create_agent(
    model=_model,
    tools=[compute_ratios, detect_inconsistencies, summarize_risks, get_context],
    system_prompt=SYSTEM_PROMPT,
)


def answer(question: str) -> str:
    result = _agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Quel est le résultat net de l'exercice ?"
    print(answer(q))
