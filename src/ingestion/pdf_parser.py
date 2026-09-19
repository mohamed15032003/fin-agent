"""
Parsing des rapports financiers PDF en texte brut structuré par page.
"""

import pymupdf  # anciennement importé "fitz" (nom encore utilisé dans la doc/tutos PyMuPDF)


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """Retourne une liste de {"page": int, "text": str} pour un PDF donné."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pages.append({"page": i + 1, "text": page.get_text()})
    return pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pdf_parser <fichier.pdf>")
        sys.exit(1)

    result = extract_text_by_page(sys.argv[1])
    print(f"{len(result)} pages extraites. Aperçu page 1:\n")
    print(result[0]["text"][:500] if result else "(vide)")
