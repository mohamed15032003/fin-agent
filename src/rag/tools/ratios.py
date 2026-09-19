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

from src.rag.tools._extraction import cite, extract_all, group_by_label, parse_chunks

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


def compute_ratios(context: str) -> str:
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
