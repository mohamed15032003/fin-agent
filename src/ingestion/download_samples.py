"""
Télécharge quelques documents financiers publics pour démarrer l'ingestion
sans attendre d'avoir résolu tous les problèmes d'accès aux données.

Sources vérifiées accessibles (voir README.md) :
- BVMT : rapports d'activité en PDF, téléchargement direct.
- SEC EDGAR : nécessite un header User-Agent identifiant (exigé par la SEC,
  ce n'est PAS une authentification payante). Voir
  https://www.sec.gov/edgar/sec-api-documentation

Usage:
    python -m src.ingestion.download_samples
"""

import os
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# TODO: ajoute d'autres rapports (par société cotée BVMT, ou d'autres années)
# via http://www.bvmt.com.tn/fr/documents/68/711/list
BVMT_SAMPLES = {
    "bvmt_rapport_activite_2024.pdf": "https://www.bvmt.com.tn/sites/default/files/rapports_activites/2024.pdf",
}

# La SEC exige un User-Agent identifiant l'appelant (nom + email), pas une clé API.
SEC_HEADERS = {"User-Agent": "TonNom ton.email@example.com"}

# Certains sites (dont potentiellement bvmt.com.tn) filtrent les requêtes sans
# User-Agent de navigateur. On en met un par défaut par précaution.
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def download(url: str, dest_path: str, headers: dict | None = None) -> None:
    print(f"Téléchargement: {url}")
    resp = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"  -> sauvegardé dans {dest_path}")


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)

    for filename, url in BVMT_SAMPLES.items():
        dest = os.path.join(RAW_DIR, filename)
        if os.path.exists(dest):
            print(f"Déjà présent, skip: {filename}")
            continue
        try:
            download(url, dest)
        except requests.RequestException as e:
            print(f"  Échec pour {filename}: {e}")

    # TODO: ajouter le téléchargement d'un ou deux 10-K via SEC EDGAR, ex:
    # https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<ticker>&type=10-K
    # puis récupérer l'URL du document depuis le JSON retourné (data.sec.gov/submissions/CIK##########.json)

    print("\nTerminé. Vérifie data/raw/ puis lance l'indexation (src/ingestion/indexer.py).")


if __name__ == "__main__":
    main()
