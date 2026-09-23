"""
Outil: résumé des facteurs de risque identifiés dans un rapport financier
(section "Facteurs de risque" / "Risk Factors").
"""

import os

from dotenv import load_dotenv
from groq import Groq
from langchain_core.tools import tool

from src.rag.retriever import retrieve

# Charge les variables du fichier .env (GROQ_API_KEY) dans l'environnement.
# Sans ca, .env peut exister et contenir la bonne cle sans que
# os.environ.get("GROQ_API_KEY") ne la voie jamais.
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# llama-3.3-70b-versatile a ete retire par Groq le 16 aout 2026 ; remplacement
# officiellement recommande par Groq (voir console.groq.com/docs/deprecations).
MODEL = "openai/gpt-oss-120b"


@tool
def summarize_risks(query: str, company: str = "") -> str:
    """Résume les principaux facteurs de risque mentionnés dans les
    rapports financiers indexés (section "Facteurs de risque" / "Risk
    Factors"). Utilise cet outil quand la question porte sur les risques,
    dangers ou menaces pesant sur l'entreprise.

    Si la question mentionne un nom d'entreprise ou de société précis
    (ex: "Délice Holding", "Société Exemple SA"), renseigne le paramètre
    company avec ce nom pour restreindre l'analyse au rapport de cette
    seule entreprise. Laisse vide si la question ne cible aucune
    entreprise en particulier."""
    hits = retrieve(query, source_filter=company if company else None)
    if company and not hits:
        return f"Aucun rapport trouvé pour l'entreprise '{company}' dans les documents indexés."

    context = "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Résume en 3 à 5 points les principaux facteurs de "
                "risque mentionnés dans le contexte fourni, extrait d'un "
                "rapport financier. Cite la source (document, page) pour "
                "chaque point.",
            },
            {"role": "user", "content": context},
        ],
    )
    return completion.choices[0].message.content
