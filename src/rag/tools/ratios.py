"""
Outil: calcul de ratios financiers clés à partir du contexte récupéré.

Approche: extraction des valeurs libellées via regex (voir `_extraction.py`),
puis calcul des ratios standards à partir des libellés disponibles. Chaque
ratio cite les valeurs et sources utilisées pour la traçabilité. Un ratio
dont un des libellés requis n'a pas été trouvé dans le contexte est signalé
comme non calculable plutôt que d'être deviné.

Limite connue: si un même libellé apparaît avec des valeurs différentes dans
le contexte (voir `inconsistency_detector.py`), ce module prend la première
valeur trouvée et le signale — ne pas se fier aveuglément au ratio dans ce
cas, vérifier d'abord la cohérence des données sources.
"""

from langchain_core.tools import tool

from src.rag.retriever import retrieve
from src.rag.tools._extraction import LABEL_DISPLAY, cite, extract_all, group_by_label, parse_chunks

# Chaque ratio: (nom affiché, formule texte, labels requis, fonction de calcul)
_RATIO_DEFINITIONS = [
    (
        "Marge nette",
        "Résultat net / Chiffre d'affaires",
        ("resultat_net", "chiffre_affaires"),
        lambda v: v["resultat_net"] / v["chiffre_affaires"] * 100,
        "%",
    ),
    (
        "ROE (Return on Equity)",
        "Résultat net / Capitaux propres",
        ("resultat_net", "capitaux_propres"),
        lambda v: v["resultat_net"] / v["capitaux_propres"] * 100,
        "%",
    ),
    (
        "Ratio d'endettement",
        "Dette financière / Capitaux propres",
        ("dette_financiere", "capitaux_propres"),
        lambda v: v["dette_financiere"] / v["capitaux_propres"],
        "x",
    ),
    (
        "Ratio de liquidité générale",
        "Actif circulant / Passif circulant",
        ("actif_circulant", "passif_circulant"),
        lambda v: v["actif_circulant"] / v["passif_circulant"],
        "x",
    ),
]


@tool
def compute_ratios(query: str, company: str = "") -> str:
    """Calcule les ratios financiers (marge nette, ROE, endettement,
    liquidité générale) à partir des rapports financiers indexés.
    Utilise cet outil quand la question porte sur des ratios,
    la rentabilité, l'endettement ou la solvabilité.

    Si la question mentionne un nom d'entreprise ou de société précis
    (ex: "Délice Holding", "Société Exemple SA"), renseigne le paramètre
    company avec ce nom pour restreindre l'analyse au rapport de cette
    seule entreprise. Laisse vide si la question ne cible aucune
    entreprise en particulier."""
    # Même logique que detect_inconsistencies (voir ce fichier et
    # rapport_sprint6.md) : ce tool sait déjà PRÉCISÉMENT quels libellés
    # financiers il doit trouver (voir _RATIO_DEFINITIONS ci-dessus et
    # LABEL_DISPLAY dans _extraction.py) — un unique retrieve(query, ...)
    # sur le texte libre de la question de l'utilisateur est donc la
    # mauvaise stratégie de recherche. On fait un retrieve() ciblé par
    # libellé connu (son nom affiché en français), avec un k plus petit
    # par appel puisque chaque requête est maintenant précise, puis on
    # fusionne les chunks récupérés en évitant les doublons. `query` n'est
    # donc plus utilisée pour le retrieve lui-même ; seul `company` reste
    # utile, pour restreindre chaque recherche par libellé à la bonne
    # source.
    source_filter = company if company else None
    seen: set[tuple] = set()
    hits = []
    for display_name in LABEL_DISPLAY.values():
        for h in retrieve(display_name, k=5, source_filter=source_filter):
            key = (h["source"], h["page"], h["text"])
            if key not in seen:
                seen.add(key)
                hits.append(h)

    if company and not hits:
        return f"Aucun rapport trouvé pour l'entreprise '{company}' dans les documents indexés."

    context = "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)

    chunks = parse_chunks(context)
    values = extract_all(chunks)
    grouped = group_by_label(values)

    if not values:
        return (
            "Aucune valeur financière libellée n'a été trouvée dans le contexte "
            "fourni (chiffre d'affaires, résultat net, capitaux propres, etc.). "
            "Impossible de calculer des ratios."
        )

    # Une valeur représentative par libellé (la première trouvée).
    picked = {label: matches[0] for label, matches in grouped.items()}

    lines = []
    for name, formula, required_labels, calc_fn, unit in _RATIO_DEFINITIONS:
        missing = [l for l in required_labels if l not in picked]
        if missing:
            lines.append(f"- {name} : non calculable (donnée manquante : {', '.join(missing)})")
            continue

        try:
            result = calc_fn({l: picked[l].value for l in required_labels})
        except ZeroDivisionError:
            lines.append(f"- {name} : non calculable (dénominateur nul)")
            continue

        citations = ", ".join(cite(picked[l]) for l in required_labels)
        ambiguous = any(len(grouped[l]) > 1 for l in required_labels)
        flag = " ⚠️ valeur ambiguë, plusieurs occurrences trouvées pour au moins un libellé — vérifier la cohérence des données" if ambiguous else ""
        lines.append(f"- {name} ({formula}) = {result:.2f}{unit} — {citations}{flag}")

    return "Ratios financiers calculés à partir du contexte fourni :\n" + "\n".join(lines)
