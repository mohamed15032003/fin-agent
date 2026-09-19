"""
Outil: détection d'incohérences chiffrées entre différentes sections d'un
même rapport (ex: un chiffre d'affaires différent entre le résumé exécutif
et les états financiers détaillés).

Approche: extraction des valeurs libellées via regex (voir `_extraction.py`),
regroupement par libellé, puis comparaison des valeurs trouvées pour un même
libellé. Si l'écart relatif dépasse une tolérance (défaut 0,5% — au-delà
d'un simple arrondi), l'incohérence est signalée avec les deux valeurs et
leurs sources (document + page) pour vérification humaine.
"""

from src.rag.tools._extraction import LABEL_DISPLAY, cite, extract_all, group_by_label, parse_chunks

# Tolérance relative en dessous de laquelle deux valeurs sont considérées
# comme le même chiffre (arrondi flottant, ex: 6.099999 vs 6.1). Volontairement
# très fine : dans un rapport financier, un écart visible à la première
# décimale (ex: 45,2 vs 45,3 millions) est presque toujours une vraie
# incohérence à signaler, pas un simple arrondi d'affichage.
_RELATIVE_TOLERANCE = 0.0005


def detect_inconsistencies(context: str) -> str:
    chunks = parse_chunks(context)
    values = extract_all(chunks)
    grouped = group_by_label(values)

    if not values:
        return (
            "Aucune valeur financière libellée n'a été trouvée dans le contexte "
            "fourni. Impossible de vérifier la cohérence des chiffres."
        )

    inconsistencies = []
    consistent_labels = []

    for label, matches in grouped.items():
        if len(matches) < 2:
            continue

        distinct = []
        for m in matches:
            if not any(abs(m.value - d.value) <= _RELATIVE_TOLERANCE * max(abs(d.value), 1) for d in distinct):
                distinct.append(m)

        display_name = LABEL_DISPLAY.get(label, label)
        if len(distinct) > 1:
            citations = " vs ".join(cite(d) for d in distinct)
            inconsistencies.append(f"- {display_name} : valeurs différentes trouvées — {citations}")
        else:
            consistent_labels.append(display_name)

    if not inconsistencies:
        detail = f" (vérifié pour : {', '.join(consistent_labels)})" if consistent_labels else ""
        return "Aucune incohérence détectée entre les occurrences trouvées dans le contexte fourni" + detail + "."

    result = "Incohérences détectées dans le contexte fourni :\n" + "\n".join(inconsistencies)
    if consistent_labels:
        result += f"\n\n(Cohérents sur les occurrences trouvées : {', '.join(consistent_labels)})"
    return result
