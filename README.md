# LO17 RAG Propriete intellectuelle

Projet Python de RAG pour aider les utilisateurs a interroger le Code de la propriete intellectuelle depuis Legifrance.

L'ancien projet JavaScript est conserve dans `src/` pour consultation. La nouvelle implementation utilise LangChain, une API OpenAI-compatible, ChromaDB et Streamlit.

## Stack

- `langchain-core`, `langchain`, `langchain-community`
- `langchain-openai` pour le LLM et les embeddings via une API OpenAI-compatible
- `chromadb` via le vector store Chroma
- `streamlit` pour l'application
- `beautifulsoup4` et `requests` pour l'ingestion Legifrance

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copier `.env.example` vers `.env`, puis renseigner `LLM_API_KEY` et `EMBEDDING_API_KEY`.

## Ingestion Legifrance

```powershell
python -m scripts.ingest_legifrance --reset
```

La commande scrape la page `LEGIFRANCE_URL`, parcourt les liens internes du Code de la propriete intellectuelle, decoupe les textes et alimente Chroma dans `chroma_db/`.

## Lancer l'application

```powershell
streamlit run streamlit_app.py
```

## Evaluation

```powershell
python -m scripts.evaluate_rag
```

Les resultats sont ecrits dans `evaluation/results/latest.json`.

L'evaluation initiale verifie des cas simples de pertinence et un cas anti-hallucination. Elle doit etre enrichie avec un jeu de questions attendu plus complet.

## Structure

- `rag_pi/config.py`: configuration et variables d'environnement
- `rag_pi/loaders.py`: scraping Legifrance
- `rag_pi/vectorstore.py`: embeddings OpenAI-compatible et Chroma
- `rag_pi/rag.py`: retrieval, prompt et generation
- `rag_pi/evaluation.py`: evaluation minimale du RAG
- `scripts/ingest_legifrance.py`: ingestion Chroma
- `scripts/evaluate_rag.py`: execution de l'evaluation
- `streamlit_app.py`: interface utilisateur

## Deploiement

Le deploiement n'est pas active pour l'instant. La cible la plus simple sera Streamlit Community Cloud ou un serveur Python classique avec les variables d'environnement configurees.
