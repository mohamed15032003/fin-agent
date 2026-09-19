"""
Outil: résumé des facteurs de risque identifiés dans un rapport financier
(section "Facteurs de risque" / "Risk Factors").
"""

import os

from dotenv import load_dotenv
from groq import Groq

# Charge les variables du fichier .env (GROQ_API_KEY) dans l'environnement.
# Sans ca, .env peut exister et contenir la bonne cle sans que
# os.environ.get("GROQ_API_KEY") ne la voie jamais.
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# llama-3.3-70b-versatile a ete retire par Groq le 16 aout 2026 ; remplacement
# officiellement recommande par Groq (voir console.groq.com/docs/deprecations).
MODEL = "openai/gpt-oss-120b"


def summarize_risks(context: str) -> str:
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
