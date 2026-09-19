"""
Génère un faux rapport financier PDF (2 pages) pour tester le pipeline
d'ingestion sans dépendre d'un téléchargement externe.

Usage:
    python tests/make_sample_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUTPUT_PATH = "data/raw/exemple_rapport_test.pdf"

PAGE_1 = [
    "Rapport Financier Annuel 2025 — Société Exemple SA",
    "",
    "Resume executif",
    "Le chiffre d'affaires de l'exercice 2025 s'eleve a 45,2 millions de dinars,",
    "en hausse de 8% par rapport a 2024 (41,9 millions de dinars).",
    "Le resultat net s'etablit a 6,1 millions de dinars.",
    "Les capitaux propres atteignent 32,4 millions de dinars.",
    "Le total du bilan s'eleve a 78,9 millions de dinars.",
]

PAGE_2 = [
    "Facteurs de risque",
    "1. Risque de change lie a l'exposition en devises etrangeres.",
    "2. Risque de concentration client (3 clients representent 40% du CA).",
    "3. Risque reglementaire lie aux evolutions fiscales locales.",
    "",
    "Etats financiers detailles",
    "Chiffre d'affaires: 45,3 millions de dinars.",  # incohérence volontaire vs page 1 (45,2)
    "Resultat net: 6,1 millions de dinars.",
    "Dette financiere nette: 12,7 millions de dinars.",
]


def make_pdf(path: str) -> None:
    c = canvas.Canvas(path, pagesize=A4)
    for page_lines in (PAGE_1, PAGE_2):
        y = 800
        for line in page_lines:
            c.drawString(50, y, line)
            y -= 20
        c.showPage()
    c.save()
    print(f"PDF de test genere: {path}")


if __name__ == "__main__":
    import os

    os.makedirs("data/raw", exist_ok=True)
    make_pdf(OUTPUT_PATH)
