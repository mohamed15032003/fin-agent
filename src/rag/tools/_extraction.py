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
    # "Total des produits d'exploitation" : libellé usuel du SCF tunisien, notamment
    # pour une holding (dividendes + management fees) qui n'a pas de "chiffre d'affaires".
    "chiffre_affaires": r"chiffre\s+d['’]affaires|total\s+des\s+produits\s+d['’]exploitation",
    # "Résultat de l'exercice" : idem, libellé courant des états financiers tunisiens.
    "resultat_net": r"r[ée]sultat\s+net|(?<!avant )r[ée]sultat\s+de\s+l['’]exercice",
    # Exclut "CAPITAUX PROPRES ET PASSIFS" (= total du bilan, pas les fonds propres)
    # et le sous-total "AVANT RESULTAT" (fonds propres hors résultat de l'exercice).
    "capitaux_propres": r"capitaux\s+propres(?!\s+et\s+passifs)(?!\s+avant\s+r[ée]sultat)",
    "total_bilan": r"total\s+(?:du\s+bilan|(?:des?\s+|de\s+l['’])?actifs?\b(?!\s+(?:non\s+)?courants?))",
    # SCF tunisien : "TOTAL DES ACTIFS/PASSIFS COURANTS" (et non "circulants"). On exige
    # le "total" : une ligne "Autres actifs/passifs courants" n'est qu'une partie du
    # total, la prendre donnerait un ratio de liquidité faux (21x au lieu de ~30x).
    "actif_circulant": r"total\s+des\s+actifs?\s+courants?|actifs?\s+circulants?",
    "passif_circulant": r"total\s+des\s+passifs?\s+courants?|passifs?\s+circulants?|dettes?\s+(?:à|a)\s+court\s+terme",
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


_NOTE_REF_PREFIX_RE = re.compile(r"[A-Za-z]-$")  # "B-8", "R-1" (renvoi à une note)
_DATE_WORD_PREFIX_RE = re.compile(r"\b(?:au|le|note)\s+$", re.IGNORECASE)  # "au 31", "Note 31"


def _is_date_or_note_ref(snippet: str, match: re.Match) -> bool:
    """Vrai si le nombre est un morceau de date ("31/12/2024") ou un renvoi de
    note ("B-8"), donc pas un montant. Sans ça, "CAPITAUX PROPRES ... Note
    31/12/2024" donnait des capitaux propres de 31 dinars (ROE aberrant)."""
    before = snippet[max(0, match.start() - 2) : match.start()]
    after = snippet[match.end() : match.end() + 1]
    if after == "/" or before.endswith("/"):
        return True
    if _NOTE_REF_PREFIX_RE.search(before):
        return True
    return bool(_DATE_WORD_PREFIX_RE.search(snippet[: match.start()]))


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
        if _is_year_like(number_match) or _is_date_or_note_ref(snippet, number_match):
            continue
        if number_match.group("magnitude") or number_match.group("currency"):
            return number_match
        if fallback is None:
            fallback = number_match
    return fallback


_WORD_START_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")


def _is_heading_only(text: str, end: int) -> bool:
    """Vrai si le libellé termine sa ligne et que la ligne suivante commence par
    un mot (ex: "CAPITAUX PROPRES" suivi de "Immobilisations incorporelles ..."):
    c'est un titre de section, et le nombre qui suit appartient à une autre ligne."""
    rest = text[end:]
    first_line_end = rest.find("\n")
    if first_line_end == -1 or rest[:first_line_end].strip():
        return False  # texte sur la même ligne, ou fin du chunk : on garde
    for line in rest[first_line_end + 1 :].split("\n"):
        line = line.strip()
        if not line or _NOTE_REF_LINE_RE.match(line):
            continue
        return bool(_WORD_START_RE.match(line))
    return False


_NOTE_REF_LINE_RE = re.compile(r"^[A-Za-z]-\d+$|^note$", re.IGNORECASE)


_DATE_YEAR_RE = re.compile(r"\b\d{1,2}/\d{1,2}/((?:19|20)\d{2})\b")


def _is_prior_period(chunk_text: str, matched_span: str) -> bool:
    """Vrai si la valeur est rattachée à une date d'exercice antérieure à la plus
    récente du chunk (colonne N-1 : "au 31/12/2023" à côté de "au 31/12/2024").
    Sans ça, le résultat 2023 était signalé comme incohérent avec celui de 2024."""
    years_in_span = [int(y) for y in _DATE_YEAR_RE.findall(matched_span)]
    if not years_in_span:
        return False
    latest = max(int(y) for y in _DATE_YEAR_RE.findall(chunk_text))
    return max(years_in_span) < latest


def extract_labeled_values(chunk: Chunk, window: int = 100) -> list[ExtractedValue]:
    """Cherche, pour chaque libellé financier connu, un nombre dans les
    `window` caractères qui suivent le libellé dans le texte du chunk."""
    results: list[ExtractedValue] = []
    text = chunk.text

    for label, label_pattern in LABEL_PATTERNS.items():
        for label_match in re.finditer(label_pattern, text, re.IGNORECASE):
            if _is_heading_only(text, label_match.end()):
                continue
            window_start = label_match.end()
            snippet = text[window_start : window_start + window]
            number_match = _best_number_in_snippet(snippet)
            if number_match is None:
                continue
            value = _parse_number(number_match)
            if value is None:
                continue
            if _is_prior_period(text, snippet[: number_match.end()]):
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
