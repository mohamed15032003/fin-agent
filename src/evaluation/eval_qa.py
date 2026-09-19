"""
Évaluation simple du pipeline RAG sur un jeu de questions/réponses de
référence, construit à la main à partir des rapports ingérés.

TODO (semaine 4):
1. Remplir GOLDEN_QA avec ~15-20 questions/réponses vérifiées manuellement
   à partir des rapports dans data/raw/.
2. Mesurer la retrieval precision (le bon chunk source est-il dans le top-k ?).
3. Mesurer la faithfulness (la réponse générée est-elle bien fondée sur le
   contexte, sans hallucination) — on peut commencer par une vérification
   manuelle, puis automatiser avec un second appel LLM "juge".
"""

from src.rag.agent import answer

GOLDEN_QA = [
    # {"question": "...", "expected_answer_contains": "...", "expected_source": "..."},
]


def run_eval() -> None:
    if not GOLDEN_QA:
        print("GOLDEN_QA est vide — ajoute des questions de référence (voir TODO).")
        return

    correct = 0
    for item in GOLDEN_QA:
        result = answer(item["question"])
        ok = item["expected_answer_contains"].lower() in result.lower()
        correct += ok
        print(f"[{'OK' if ok else 'FAIL'}] {item['question']}")

    print(f"\nScore: {correct}/{len(GOLDEN_QA)}")


if __name__ == "__main__":
    run_eval()
