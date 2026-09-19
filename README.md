# FinAgent — Agent IA pour l'Analyse de Documents Financiers

> Agent IA basé sur une architecture **RAG (Retrieval-Augmented Generation)** pour l'analyse automatisée de rapports financiers publics (rapports annuels BVMT, 10-K) : extraction de données, détection d'incohérences, calcul de ratios financiers et résumé des facteurs de risque — avec une couche **Model Context Protocol (MCP)** pour une intégration standardisée.
>
> Projet inspiré des initiatives GenAI actuelles des Big Four dans l'audit augmenté par IA ([EY.ai](https://chatfin.ai/blog/big-4-ai-agents-ey-kpmg-deloitte-pwc-finance-teams-2026/), PwC GL.ai / ChatPwC).

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Roadmap](#roadmap)
- [Évaluation](#évaluation)

## Statut de validation (mise à jour semaine 3)

Ce qui a été testé et fonctionne réellement (pas juste écrit) :

- ✅ Parsing PDF (`pdf_parser.py`) — extrait correctement le texte page par page.
- ✅ Chunking (`chunker.py`) — découpe le texte en chunks avec métadonnées.
- ✅ Mécanique ChromaDB (add / persist / query) — validée avec un embedding factice.
- ✅ Tous les modules s'importent sans déclencher d'appel réseau caché (le chargement du modèle d'embeddings est maintenant paresseux, voir `get_embedding_fn()` dans `indexer.py`/`retriever.py`).
- ✅ L'API FastAPI démarre et répond sur `/health`.
- ✅ Outil ratios financiers (`ratios.py`) — implémenté (extraction par regex + calcul), testé sur le rapport d'exemple : marge nette, ROE et ratio d'endettement corrects ; signale proprement les ratios non calculables (donnée absente) plutôt que d'inventer un chiffre.
- ✅ Outil détection d'incohérences (`inconsistency_detector.py`) — implémenté, testé : détecte bien l'écart volontaire chiffre d'affaires 45,2 vs 45,3 millions entre la page 1 et la page 2 du rapport d'exemple, sans faux positif sur les valeurs identiques (résultat net).
- ✅ Outil résumé de risques (`risk_summarizer.py`) — déjà implémenté (appel direct à l'API Groq), pas encore testé avec une vraie clé API.
- ✅ Serveur MCP (`mcp/server.py`) — était déjà écrit mais utilisait l'ancienne API v1 du SDK (`FastMCP`) alors que `pip install` installe la v2 ; le SDK a renommé `FastMCP` en `MCPServer` (déplacé vers `mcp.server.mcpserver`), ce qui cassait l'import. Corrigé + requirements.txt épinglé sur `mcp>=2.0`. Les imports sont validés, l'appel réel des tools MCP (nécessite Groq) reste à tester.

Ce qui n'a **toujours pas** pu être testé depuis les environnements de développement utilisés jusqu'ici (réseau restreint des deux côtés), à tester toi-même :

- Le téléchargement du modèle d'embeddings `all-MiniLM-L6-v2` depuis huggingface.co (nécessite juste une connexion internet normale — devrait fonctionner sans problème depuis ta machine).
- Le téléchargement des rapports BVMT (`download_samples.py`) — l'URL est bonne (trouvée sur le site officiel), mais si `bvmt.com.tn` bloque encore après ça, utilise en attendant `tests/make_sample_pdf.py` (génère un faux rapport de test) ou une des sources de secours (SEC EDGAR, Kaggle) listées plus bas.
- L'appel réel à l'API Groq, donc `agent.py`, `risk_summarizer.py` et le routage par mots-clés bout en bout (nécessite ta clé dans `.env`) — **c'est le prochain test à faire en priorité**, tout le reste en dépend (semaine 2 du planning initial, toujours pas franchie faute de réseau des deux côtés).

Pour démarrer immédiatement sans attendre d'avoir résolu l'accès aux données :

```bash
python tests/make_sample_pdf.py   # génère data/raw/exemple_rapport_test.pdf
python -m src.ingestion.indexer   # indexe ce PDF de test
python -m src.rag.retriever "risque de change"   # teste la recherche

# une fois GROQ_API_KEY renseignée dans .env :
python -m src.rag.agent "quels sont les ratios financiers ?"
python -m src.rag.agent "y a-t-il des incoherences dans le rapport ?"
```

## Fonctionnalités

- [x] Ingestion et parsing de rapports financiers PDF
- [x] Indexation vectorielle (RAG) pour la recherche sémantique dans les documents — mécanique validée, modèle d'embeddings pas encore téléchargé en conditions réelles
- [ ] Question-réponse sur le contenu d'un rapport — code écrit (`agent.py`), pas encore testé avec une vraie clé Groq
- [x] Calcul automatique de ratios financiers clés
- [x] Détection d'incohérences chiffrées entre sections d'un même rapport
- [x] Résumé des facteurs de risque identifiés — code écrit, pas encore testé avec une vraie clé Groq
- [x] Exposition des outils de l'agent via un serveur MCP — implémenté (`mcp/server.py`), imports vérifiés
- [ ] Interface web simple pour la démo

## Architecture

```
Documents PDF (BVMT / 10-K)
        │
        ▼
  Ingestion & parsing  ──►  Chunking & embeddings  ──►  ChromaDB (index vectoriel)
                                                              │
                                                              ▼
                                                     Agent orchestrateur (LangChain + Groq LLM)
                                                              │
                                    ┌─────────────────────────┼─────────────────────────┐
                                    ▼                         ▼                         ▼
                          Outil: ratios financiers   Outil: détection          Outil: résumé
                                                      d'incohérences            de risques
                                                              │
                                                              ▼
                                                     Serveur MCP (expose les outils)
                                                              │
                                                              ▼
                                                   API FastAPI ──► Interface (Streamlit/React)
```

## Stack technique

| Composant | Technologie |
|---|---|
| Parsing PDF | PyMuPDF / pdfplumber |
| Embeddings & vector store | ChromaDB |
| Orchestration agent | LangChain |
| LLM | API Groq (llama-3.3-70b-versatile) |
| Backend | FastAPI |
| Interface | Streamlit (MVP) → React/TypeScript (évolution) |
| Protocole d'intégration | Model Context Protocol (MCP) |

## Installation

```bash
git clone <ton-repo>
cd fin-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # renseigner GROQ_API_KEY
```

## Ouvrir dans IntelliJ IDEA

1. **Plugin Python** : Ultimate l'a nativement ; en Community, va dans `Settings > Plugins > Marketplace`, cherche "Python" et installe le plugin (fonctionnalités un peu réduites par rapport à Ultimate/PyCharm, mais suffisant pour ce projet).
2. `File > Open`, sélectionne le dossier `fin-agent/`.
3. Crée l'environnement virtuel si ce n'est pas déjà fait (`python -m venv venv` à la racine), puis `File > Project Structure > SDKs > + > Add Python SDK > Existing environment`, pointe vers `venv/bin/python` (ou `venv\Scripts\python.exe` sous Windows).
4. Clic droit sur `src/` → `Mark Directory as > Sources Root` (pour que les imports `from src....` soient résolus correctement par l'IDE).
5. Crée des **Run Configurations** (`Run > Edit Configurations > +`) pour aller plus vite :
   - `Python` : module `src.ingestion.download_samples`
   - `Python` : module `src.ingestion.indexer`
   - `Python` : module `src.rag.agent`, arguments = ta question de test
   - `FastAPI`/`Python` : script `uvicorn`, arguments `src.api.main:app --reload`
6. Copie `.env.example` en `.env` à la racine — IntelliJ (avec le plugin EnvFile, optionnel) peut le charger automatiquement dans les Run Configurations, sinon exporte les variables manuellement dans ton shell avant de lancer.

## Utilisation

```bash
# 1. Déposer des rapports financiers PDF dans data/raw/

# 2. Lancer l'ingestion
python -m src.ingestion.indexer

# 3. Lancer l'API
uvicorn src.api.main:app --reload

# 4. (bonus) Lancer le serveur MCP
python -m src.mcp.server
```

## Roadmap

- **Semaine 1** — Ingestion & données : parsing, chunking, indexation ChromaDB sur 5-10 rapports. ✅ Mécanique validée sur un rapport de test ; téléchargement du modèle d'embeddings et ingestion de vrais rapports BVMT encore à faire (bloqué par le réseau des environnements de dev utilisés jusqu'ici — à tester depuis ta machine).
- **Semaine 2** — RAG de base : question-réponse fonctionnel via LangChain + Groq. ⏳ Code écrit (`agent.py`, `retriever.py`), **jamais testé avec une vraie clé Groq** — c'est le prochain blocage à lever en priorité, avant de pouvoir valider tout le reste bout en bout.
- **Semaine 3** — Outils spécialisés + interface : ratios, incohérences, résumé de risques, démo Streamlit/FastAPI. ✅ Ratios et incohérences implémentés et testés (regex, sans dépendre de Groq) ; résumé de risques écrit mais pas testé (dépend de Groq) ; interface de démo pas commencée.
- **Semaine 4** — Couche MCP, évaluation, déploiement (Render/Railway), README final, post LinkedIn. ✅ Couche MCP implémentée (et un bug d'API v1/v2 du SDK corrigé) ; évaluation, déploiement et post LinkedIn pas commencés.

## Évaluation

Jeu de questions/réponses de référence sur les rapports ingérés (voir `src/evaluation/eval_qa.py`), mesurant :
- **Retrieval precision** : les bons passages sont-ils récupérés ?
- **Faithfulness** : la réponse générée est-elle fidèle aux documents sources (pas d'hallucination) ?

## Sources de données (vérifiées accessibles)

| Source | Accès | Format | Usage |
|---|---|---|---|
| [BVMT — Rapports d'activité](https://www.bvmt.com.tn/fr/rapports-activites) | Public, PDF téléchargeable directement (ex: [rapport 2024](https://www.bvmt.com.tn/sites/default/files/rapports_activites/2024.pdf)) | PDF | Contexte marché tunisien, cas d'usage local |
| [BVMT — Documents / Rapports annuels de gestion](http://www.bvmt.com.tn/fr/documents/68/711/list) | Public, liste de PDF par société cotée | PDF | Rapports par société, texte pour le RAG |
| [SEC EDGAR Full-Text Search](https://www.sec.gov/cgi-bin/srqsb?text=...) / [data.sec.gov](https://www.sec.gov/edgar/sec-api-documentation) | 100% gratuit, sans clé API — juste un header `User-Agent` identifiant (ex: `"TonNom ton@email.com"`, exigé par la SEC, pas une authentification payante) | PDF/HTML/JSON | Rapports 10-K/10-Q en anglais, volume important |
| [SEC Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets.html) | Gratuit, officiel, jeux de données trimestriels structurés (pas du PDF) | TSV/JSON | Calcul de ratios sans avoir à parser du texte — complète bien les PDF pour la partie "chiffrée" |
| [Kaggle — Financial Statements datasets](https://www.kaggle.com/datasets?search=financial+statements) | Gratuit, compte Kaggle requis | CSV | Solution de secours si le scraping BVMT/EDGAR bloque en semaine 1 |

`src/ingestion/download_samples.py` télécharge automatiquement quelques documents de test depuis ces sources pour démarrer l'ingestion dès le premier jour, sans attendre d'avoir résolu l'accès aux données.

## Mots-clés

RAG · LLM Agents · LangChain · ChromaDB · Model Context Protocol (MCP) · FastAPI · Financial Document Analysis · Audit Automation
