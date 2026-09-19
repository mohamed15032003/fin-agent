# Architecture détaillée

Voir le schéma dans le README.md principal.

## Décisions techniques et justifications

| Décision | Justification |
|---|---|
| ChromaDB comme vector store | Déjà utilisé sur le projet FORSA — pas de courbe d'apprentissage |
| LangChain + Groq (llama-3.3-70b) | Déjà maîtrisé (chatbot RBAC FORSA) |
| FastAPI pour le backend | Stack déjà maîtrisé (MedScraping) |
| MCP pour exposer les outils | Différenciant vs. un simple projet RAG ; standard émergent dans l'IA agentique 2026 |
| Sources publiques (BVMT, SEC EDGAR) | Évite la dépendance à des données bancaires tunisiennes difficiles d'accès |

## Points d'attention identifiés

- Les rapports BVMT sont en français : vérifier que le modèle d'embeddings
  choisi (actuellement `all-MiniLM-L6-v2`, anglophone) fonctionne bien, sinon
  basculer sur un modèle multilingue (ex: `paraphrase-multilingual-mpnet-base-v2`).
- La détection d'incohérences et le calcul de ratios nécessitent une
  extraction fiable des valeurs numériques : à ne pas laisser reposer
  uniquement sur la génération libre du LLM (risque d'hallucination de chiffres).
