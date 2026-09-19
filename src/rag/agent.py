"""
Agent orchestrateur : reçoit une question, récupère le contexte pertinent (RAG),
décide s'il doit appeler un outil spécialisé (ratios, incohérences, résumé de
risques), puis génère une réponse via l'API Groq.

TODO (semaine 2-3): remplacer cette boucle simple par un vrai agent LangChain
(AgentExecutor + tool calling natif de Groq/Llama) une fois le RAG de base validé.
"""

import os
import unicodedata

from groq import Groq

from src.rag.retriever import retrieve
from src.rag.tools.ratios import compute_ratios
from src.rag.tools.inconsistency_detector import detect_inconsistencies
from src.rag.tools.risk_summarizer import summarize_risks

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# llama-3.3-70b-versatile a ete retire par Groq le 16 aout 2026 ; remplacement
# officiellement recommande par Groq (voir console.groq.com/docs/deprecations).
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """Tu es un assistant d'analyse financière. Réponds uniquement
à partir du contexte fourni, extrait de rapports financiers. Si l'information
n'est pas dans le contexte, dis-le clairement plutôt que d'inventer."""

# TODO (semaine 3): remplacer ce routage par mots-clés par un vrai tool-calling
# structuré (function calling) supporté par l'API Groq.
TOOLS = {
    "ratios": compute_ratios,
    "incoherences": detect_inconsistencies,  # comparé à la question sans accents, voir _normalize()
    "risques": summarize_risks,
}


def _normalize(text: str) -> str:
    """Minuscules + accents retirés, pour que le routage par mots-clés marche
    que la question contienne "incohérences" ou "incoherences"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def answer(question: str) -> str:
    hits = retrieve(question)
    context = "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)

    normalized_question = _normalize(question)
    for keyword, tool_fn in TOOLS.items():
        if keyword in normalized_question:
            return tool_fn(context)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Quel est le résultat net de l'exercice ?"
    print(answer(q))
