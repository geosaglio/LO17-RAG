# LO17 RAG Propriété intellectuelle

Assistant RAG pour interroger le Code de la propriété intellectuelle via Legifrance.

Ce projet permet de :
- ingérer les textes du Code depuis Legifrance
- indexer les documents dans Chroma
- utiliser un LLM OpenAI-compatible pour répondre aux questions
- afficher les résultats dans Streamlit
- évaluer le système avec un jeu de cas structuré
- scorer et reranker les documents pertinents

## Nettoyage et choix projet

Le projet est maintenant épuré :
- seuls les composants utiles sont conservés
- les scripts d'exploration inutilisés ont été supprimés
- la logique RAG, l'évaluation et l'UI restent au centre
- le notebook externe n'est pas conservé dans la version de soutenance

## Dépendances

Installer l'environnement et les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Copier `.env.example` vers `.env` et renseigner les variables :

- `LLM_BASE_URL` : endpoint UTC API, par exemple `https://ia.beta.utc.fr/api`
- `LLM_API_KEY` : clé UTC pour le modèle
- `EMBEDDING_BASE_URL` : endpoint embeddings
- `EMBEDDING_API_KEY` : clé embeddings
- `LLM_MODEL` : modèle à utiliser, ex. `RedHatAI/gemma-4-31B-it-NVFP4`

Paramètres RAG importants :

- `CHUNK_SIZE` : taille d'un segment de texte
- `CHUNK_OVERLAP` : chevauchement entre segments
- `RETRIEVER_K` : nombre de documents récupérés
- `MIN_RELEVANCE_SCORE` : seuil minimal de pertinence
- `RERANK_OVERLAP_WEIGHT` : bonus de reranking lexical

## Commandes principales

### Ingestion des données

```powershell
python -m scripts.ingest_legifrance --reset
```

### Lancer l'interface

```powershell
streamlit run streamlit_app.py
```

### Évaluation

```powershell
python -m scripts.evaluate_rag
```

Les résultats sont enregistrés dans :
- `evaluation/results/latest.json`
- `evaluation/results/latest.csv`

## Scoring et évaluation

Le pipeline utilise :
- des embeddings OpenAI-compatible
- le vectorstore Chroma
- un reranking explicite basé sur la similarité et l'overlap lexical
- un seuil de pertinence configurable
- une évaluation structurée de cas factuels et de refus

### Résultats d'évaluation

- supervision par termes attendus
- reconnaissance des sources citées
- détection des réponses de refus/hallucination
- mesures de grounding et de correspondance

## Architecture du projet

- `rag_pi/config.py` : paramètres de configuration et variables d'environnement
- `rag_pi/loaders.py` : scraping et parsing Legifrance
- `rag_pi/vectorstore.py` : construction du store Chroma et client d'embeddings
- `rag_pi/rag.py` : récupération, reranking et génération de réponses
- `rag_pi/prompts.py` : prompt RAG spécifique au domaine juridique
- `rag_pi/evaluation_cases.py` : définition des cas d'évaluation
- `rag_pi/evaluation.py` : pipeline d'évaluation et scoring
- `scripts/ingest_legifrance.py` : ingestion des documents dans Chroma
- `scripts/evaluate_rag.py` : exécution de l'évaluation
- `streamlit_app.py` : interface utilisateur Streamlit

## À présenter en soutenance

- architecture claire et modulaire
- intégration à l'API UTC pour LLM et embeddings
- scoring des documents et reranking visible côté UI
- évaluation automatisée et exportable JSON/CSV
- test de refus/hallucination pour éviter les réponses inventées

## Notes

- Le projet se concentre sur un pipeline RAG réaliste et compréhensible.
- Les composants superflus ont été retirés pour garder un dossier propre.
