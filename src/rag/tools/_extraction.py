"""
Utilitaires partagés par les outils d'analyse (ratios, incohérences) :
- découpage du `context` (tel que construit par src/rag/agent.py) en chunks
  avec leur source/page d'origine, pour garder la traçabilité ;
- extraction de couples (libellé financier, valeur numérique) à partir de
  texte français de rapport financier, via des expressions régulières.

Reste volontairement simple (regex) plutôt que d'appeler le LLM pour
extraire les chiffres : moins cher, déterministe, et suffisant pour des
rapports qui suivent un vocabulaire assez standard (BVMT/10-K). Peut être
remplacé plus tard par une extraction structurée via function calling Groq
si les regex s'avèrent trop fragiles sur de vrais rapports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Découpage du contexte en chunks (source, page, texte)
# ---------------------------------------------------------------------------

# Format produit par src/rag/agent.py: "[<source> p.<page>] <texte>", chunks
# séparés par une ligne vide.
_CHUNK_HEADER_RE = re.compile(r"^\[(?P<source>[^\]]+)\s+p\.(?P<page>\S+)\]\s?(?P<text>.*)$", re.DOTALL)


@dataclass
class Chunk:
    source: str | None
    page: str | None
    text: str


def parse_chunks(context: str) -> list[Chunk]:
    """Reconstitue les chunks (avec source/page) à partir du `context` fourni
    aux outils. Si le contexte n'a pas le format attendu (ex: appelé dans un
    test avec du texte brut), retourne un chunk unique sans source connue."""
    raw_chunks = [c for c in context.split("\n\n") if c.strip()]
    if not raw_chunks:
        return []

    parsed = []
    for raw in raw_chunks:
        m = _CHUNK_HEADER_RE.match(raw.strip())
        if m:
            parsed.append(Chunk(source=m.group("source"), page=m.group("page"), text=m.group("text")))
        else:
            parsed.append(Chunk(source=None, page=None, text=raw))
    return parsed


# ---------------------------------------------------------------------------
# 2. Extraction de valeurs financières libellées
# ---------------------------------------------------------------------------

# Libellé -> clé canonique. L'ordre compte un peu (les plus spécifiques
# d'abord) pour éviter qu'un libellé générique n'avale un libellé précis.
LABEL_PATTERNS: dict[str, str] = {
    "dette_financiere": r"dettes?\s+financi[eè]res?(?:\s+nettes?)?",
    "chiffre_affaires": r"chiffre\s+d['’]affaires",
    "resultat_net": r"r[ée]sultat\s+net",
    "capitaux_propres": r"capitaux\s+propres",
    "total_bilan": r"total\s+(?:du\s+bilan|(?:de\s+l['’])?actif)",
    "actif_circulant": r"actifs?\s+circulants?",
    "passif_circulant": r"passifs?\s+circulants?|dettes?\s+(?:à|a)\s+court\s+terme",
}

# Libellé lisible pour l'affichage des résultats.
LABEL_DISPLAY = {
    "dette_financiere": "Dette financière (nette)",
    "chiffre_affaires": "Chiffre d'affaires",
    "resultat_net": "Résultat net",
    "capitaux_propres": "Capitaux propres",
    "total_bilan": "Total du bilan",
    "actif_circulant": "Actif circulant",
    "passif_circulant": "Passif circulant",
}

# Nombre français ou anglais : "45,2" / "45.2" / "1 234,5", suivi
# optionnellement d'une unité de grandeur (million/millier) et d'une devise.
# NB: l'alternative "avec séparateur de milliers" exige au moins UN groupe
# "espace + 3 chiffres" (le "+" est important) — sinon un nombre simple comme
# "2025" se ferait tronquer en "202" par le premier groupe de 1-3 chiffres.
_NUMBER_RE = (
    r"(?P<value>\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    r"\s*(?P<magnitude>millions?|milliers?|mille|k)?\s*"
    r"(?:de\s+)?"
    r"(?P<currency>dinars?|dt|mdt|md|tnd|€|eur|usd|\$)?"
)

# Un nombre à 4 chiffres sans unité ni devise, dans une plage d'années
# plausible, est presque toujours une date ("exercice 2025") et non une
# valeur financière — on l'ignore pour éviter de la prendre comme montant.
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _is_year_like(match: re.Match) -> bool:
    if match.group("magnitude") or match.group("currency"):
        return False
    cleaned = (match.group("value") or "").replace(" ", "").replace(" ", "")
    return bool(_YEAR_RE.fullmatch(cleaned))

_MAGNITUDE_MULTIPLIER = {
    "million": 1_000_000,
    "millions": 1_000_000,
    "millier": 1_000,
    "milliers": 1_000,
    "mille": 1_000,
    "k": 1_000,
}


@dataclass
class ExtractedValue:
    label: str  # clé canonique, ex: "chiffre_affaires"
    value: float  # valeur normalisée (unité de base, ex: dinars)
    raw: str  # texte source de la valeur, ex: "45,2 millions de dinars"
    source: str | None
    page: str | None

    def formatted(self) -> str:
        # Ré-affiche en millions pour la lisibilité si la valeur est grande.
        if abs(self.value) >= 1_000_000:
            return f"{self.value / 1_000_000:.2f} M"
        if abs(self.value) >= 1_000:
            return f"{self.value / 1_000:.2f} K"
        return f"{self.value:.2f}"


def _parse_number(match: re.Match) -> float | None:
    raw_value = match.group("value")
    if raw_value is None:
        return None
    # Nombre français: "1 234,5" -> enlève les espaces (séparateur de
    # milliers), remplace la virgule décimale par un point.
    cleaned = raw_value.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    magnitude = (match.group("magnitude") or "").lower()
    return value * _MAGNITUDE_MULTIPLIER.get(magnitude, 1)


def _best_number_in_snippet(snippet: str) -> re.Match | None:
    """Parmi tous les nombres trouvés dans le snippet, choisit le plus
    plausible comme valeur financière : priorité au premier nombre assorti
    d'une unité de grandeur ou d'une devise (ex: "45,2 millions de dinars"),
    en ignorant les nombres qui ressemblent à une année ("exercice 2025").
    À défaut d'un tel candidat, retombe sur le premier nombre non-année."""
    fallback: re.Match | None = None
    for number_match in re.finditer(_NUMBER_RE, snippet, re.IGNORECASE):
        if _parse_number(number_match) is None:
            continue
        if _is_year_like(number_match):
            continue
        if number_match.group("magnitude") or number_match.group("currency"):
            return number_match
        if fallback is None:
            fallback = number_match
    return fallback


def extract_labeled_values(chunk: Chunk, window: int = 100) -> list[ExtractedValue]:
    """Cherche, pour chaque libellé financier connu, un nombre dans les
    `window` caractères qui suivent le libellé dans le texte du chunk."""
    results: list[ExtractedValue] = []
    text = chunk.text

    for label, label_pattern in LABEL_PATTERNS.items():
        for label_match in re.finditer(label_pattern, text, re.IGNORECASE):
            window_start = label_match.end()
            snippet = text[window_start : window_start + window]
            number_match = _best_number_in_snippet(snippet)
            if number_match is None:
                continue
            value = _parse_number(number_match)
            if value is None:
                continue
            raw = (label_match.group(0) + snippet[: number_match.end()]).strip()
            raw = " ".join(raw.split())  # normalise les espaces/retours à la ligne
            results.append(
                ExtractedValue(label=label, value=value, raw=raw, source=chunk.source, page=chunk.page)
            )

    return results


def extract_all(chunks: list[Chunk]) -> list[ExtractedValue]:
    values: list[ExtractedValue] = []
    for chunk in chunks:
        values.extend(extract_labeled_values(chunk))
    return values


def group_by_label(values: list[ExtractedValue]) -> dict[str, list[ExtractedValue]]:
    grouped: dict[str, list[ExtractedValue]] = {}
    for v in values:
        grouped.setdefault(v.label, []).append(v)
    return grouped


def cite(v: ExtractedValue) -> str:
    if v.source and v.page:
        return f"{v.raw} ({v.source} p.{v.page})"
    return v.raw
