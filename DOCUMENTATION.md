# Documentation du projet — LO17 RAG Propriété intellectuelle

Ce document décrit le fonctionnement global du projet puis détaille chacun de ses
composants, en mettant en évidence les mesures mises en place pour améliorer
la performance du système RAG (Retrieval-Augmented Generation).

Le projet est une application Streamlit qui répond à des questions sur le
**Code de la propriété intellectuelle** français à partir de textes scrappés
sur **Legifrance**, indexés dans **Chroma**, et interrogés via un **LLM
OpenAI-compatible** hébergé par l'UTC.

---

## 1. Vue d'ensemble

### 1.1. Pile technique

| Brique             | Choix                                                          |
|--------------------|----------------------------------------------------------------|
| Source de données  | Legifrance — `LEGITEXT000006069414` (Code de la PI)            |
| Crawler            | `requests` + `BeautifulSoup`                                   |
| Découpage          | `RecursiveCharacterTextSplitter` (LangChain)                   |
| Embeddings         | API OpenAI-compatible (UTC OpenWebUI / Ollama)                 |
| Vectorstore        | Chroma persistant local                                        |
| Orchestration      | LangChain (`ChatOpenAI`, `ChatPromptTemplate`)                 |
| LLM                | Modèle UTC (ex. `mistral-small3.2:latest`)                     |
| Interface          | Streamlit                                                      |
| Évaluation         | Pipeline maison avec cas typés (factuel / multi-hop / refus)   |

### 1.2. Schéma général de l'architecture

```
                       ┌─────────────────────────────┐
                       │       Legifrance (HTML)     │
                       │  Code de la PI : LEGITEXT…  │
                       └──────────────┬──────────────┘
                                      │ requests + BS4
                                      ▼
                       ┌─────────────────────────────┐
                       │  rag_pi/loaders.py          │
                       │  crawl_legifrance_code()    │
                       │  → Documents LangChain      │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │  scripts/ingest_legifrance  │
                       │  RecursiveCharacterSplitter │
                       │  + batchs d'embeddings      │
                       └──────────────┬──────────────┘
                                      │ vecteurs + metadata
                                      ▼
                       ┌─────────────────────────────┐
                       │      Chroma (persistant)    │
                       │  collection : legifrance_*  │
                       └──────────────┬──────────────┘
                                      │
        ┌─────────────────────────────┴──────────────────────────────┐
        │                                                            │
        ▼                                                            ▼
┌──────────────────────┐                              ┌──────────────────────────┐
│  streamlit_app.py    │                              │  scripts/evaluate_rag.py │
│   chat utilisateur   │                              │  cas → métriques JSON/CSV│
└──────────┬───────────┘                              └──────────┬───────────────┘
           │  question                                           │
           ▼                                                     ▼
                    ┌──────────────────────────────────┐
                    │       rag_pi/rag.py              │
                    │   answer_question(question)      │
                    │  ┌────────────────────────────┐  │
                    │  │ 1. similarity_search_score │  │
                    │  │ 2. rerank lexical overlap  │  │
                    │  │ 3. filtre min_score        │  │
                    │  │ 4. prompt + LLM (chain)    │  │
                    │  └────────────────────────────┘  │
                    └───────────────┬──────────────────┘
                                    │ RagAnswer
                                    ▼
                ┌────────────────────────────────────────┐
                │  réponse + sources + scores affichés   │
                └────────────────────────────────────────┘
```

### 1.3. Cycle de vie d'une question (séquence)

```
Utilisateur          Streamlit          rag.py            Chroma           LLM (UTC)
   │   question         │                  │                  │                │
   │ ─────────────────► │                  │                  │                │
   │                    │ answer_question()│                  │                │
   │                    │ ───────────────► │                  │                │
   │                    │                  │ load_vectorstore │                │
   │                    │                  │ ───────────────► │                │
   │                    │                  │ similarity_search_with_score      │
   │                    │                  │ ───────────────► │                │
   │                    │                  │ ◄─── docs+scores │                │
   │                    │                  │                  │                │
   │                    │                  │ rerank lexical   │                │
   │                    │                  │ filtre min_score │                │
   │                    │                  │                  │                │
   │                    │                  │ format_docs + RAG_PROMPT          │
   │                    │                  │ ───────────────────────────────►  │
   │                    │                  │ ◄─── réponse texte ────────────── │
   │                    │ ◄── RagAnswer ── │                  │                │
   │ ◄── réponse + sources + scores        │                  │                │
```

### 1.4. Cartographie des fichiers

```
LO17_RAG/
├── streamlit_app.py             ← UI Streamlit
├── README.md                    ← README de soutenance
├── DOCUMENTATION.md             ← (ce fichier)
├── requirements.txt / pyproject.toml
├── .env.example                 ← variables d'environnement
│
├── rag_pi/                      ← package principal
│   ├── __init__.py
│   ├── config.py                ← Settings + load_settings()
│   ├── loaders.py               ← crawler Legifrance → Documents
│   ├── vectorstore.py           ← embeddings + Chroma
│   ├── prompts.py               ← prompt RAG juridique
│   ├── rag.py                   ← retrieve + rerank + génération
│   ├── evaluation.py            ← pipeline d'évaluation
│   └── evaluation_cases.py      ← cas de test
│
├── scripts/
│   ├── ingest_legifrance.py     ← peuple la collection Chroma
│   └── evaluate_rag.py          ← lance l'évaluation, exporte JSON/CSV
│
└── Rapport/                     ← rapport LaTeX (hors pipeline)
```

---

## 2. Détail de chaque partie et leviers de performance

Chaque sous-section décrit le rôle du module, son fonctionnement, puis
les **leviers spécifiques de performance** mis en place pour rendre le RAG
plus pertinent, plus robuste et moins sujet aux hallucinations.

### 2.1. Configuration centralisée — `rag_pi/config.py`

Le module définit une dataclass `Settings` frozen, peuplée depuis le fichier
`.env` via `python-dotenv`. Toutes les variables critiques (URLs, clés API,
modèles, hyperparamètres RAG) y sont regroupées et valident leur présence au
démarrage (`RuntimeError` si une variable est manquante).

**Leviers de performance :**

- **Hyperparamètres exposés en clair** : `chunk_size`, `chunk_overlap`,
  `retriever_k`, `min_relevance_score`, `rerank_overlap_weight` sont
  réglables sans toucher au code, ce qui facilite le tuning empirique.
- **Robustesse réseau paramétrable** : `embedding_batch_size`,
  `embedding_sleep_seconds`, `embedding_max_retries` permettent d'ajuster
  le débit d'appels à l'API d'embeddings (très utile face à un endpoint
  partagé qui peut throttler).
- **Échec rapide à la configuration** : la vérification des variables
  obligatoires évite des erreurs silencieuses à mi-pipeline.

### 2.2. Ingestion des sources — `rag_pi/loaders.py`

Le module expose `crawl_legifrance_code(start_url, max_pages)` qui réalise
un BFS borné sur les pages internes de Legifrance liées au
`LEGITEXT000006069414` (Code de la PI). Chaque article (sélecteur
`article.js-article-content-to-copy`) est transformé en `Document`
LangChain avec un texte propre (titre + badge + date + contenu) et des
métadonnées (URL ancrée, titre de page, `article_id`, `article_title`).

**Leviers de performance :**

- **BFS borné par `max_pages`** : on évite l'explosion combinatoire du
  graphe Legifrance tout en restant exhaustif sur la portion utile.
- **Filtrage strict des liens** : `_legifrance_links()` ne suit que les
  URLs qui restent dans le bon `LEGITEXT`, ce qui empêche de polluer
  l'index avec d'autres codes.
- **Métadonnées riches conservées dans le store** : `article_id`,
  `article_title`, `source` (URL ancrée). Ces métadonnées sont exploitées
  plus tard pour :
  1. afficher des liens précis vers l'article cité,
  2. enrichir le calcul de recouvrement lexical (voir §2.4),
  3. permettre l'évaluation par identifiant d'article (`L112-4`, …).
- **Déduplication par `article_id` et par contenu** (`seen_article_ids`,
  `seen`) : limite la redondance dans l'index et évite que plusieurs
  chunks quasi-identiques saturent le top-k de la recherche.
- **Fallback de contenu** : `_should_keep_fallback_document` conserve une
  page comme document de secours quand aucun article n'a été extrait,
  uniquement si l'URL n'est pas une simple table des matières — utile pour
  ne pas perdre les sections explicatives sans pour autant indexer du
  bruit.
- **Nettoyage HTML** : suppression des balises `script/style/noscript/svg`
  et normalisation des espaces dans `_clean_text`, pour un texte plus
  propre côté embeddings.

### 2.3. Indexation vectorielle — `rag_pi/vectorstore.py` et `scripts/ingest_legifrance.py`

`vectorstore.py` définit un client d'embeddings maison
`SingleTextOpenAICompatibleEmbeddings` capable de parler plusieurs
dialectes (`openai`, `openwebui`, `ollama_embed`, `ollama_embeddings`),
puis construit le store Chroma persistant via `load_vectorstore()`.

`scripts/ingest_legifrance.py` orchestre :
1. le crawl Legifrance,
2. le split en chunks `chunk_size=1200`, `chunk_overlap=180`,
3. l'embedding par lots avec retry,
4. l'écriture dans la collection Chroma (avec option `--reset`).

```
crawl_legifrance_code() ──► [Documents bruts]
        │
        ▼
RecursiveCharacterTextSplitter(1200, 180)
        │
        ▼
boucle batchs (taille embedding_batch_size)
   ├── embed_documents_with_retry()  ← backoff exponentiel
   └── vectorstore._collection.add(ids, embeddings, documents, metadatas)
        │
        ▼
    Chroma persistant (./chroma_db/)
```

**Leviers de performance :**

- **Chunking récursif avec chevauchement** : `chunk_size=1200`,
  `chunk_overlap=180` (séparateurs `["\n\n", "\n", ". ", " ", ""]`). Le
  chevauchement évite qu'une définition d'article soit coupée pile à la
  frontière entre deux chunks, ce qui dégraderait le rappel.
- **IDs stables** (`_stable_id`) basés sur `article_id` + index : un
  ré-import ne duplique pas les passages, et `--reset` repart d'une base
  saine.
- **Embeddings par lots** (`embedding_batch_size`) avec **sleep entre
  lots** (`embedding_sleep_seconds`) : limite la pression sur l'API
  partagée UTC et réduit les `429`.
- **Retry exponentiel** (`embed_documents_with_retry`) : `base_sleep * 2^n`
  jusqu'à `embedding_max_retries`, sur erreurs réseau (`RequestException`)
  — assure une ingestion robuste sur un endpoint peu fiable.
- **Multi-format embeddings** : la classe sait gérer un endpoint
  OpenWebUI, OpenAI strict et Ollama (deux variantes). Permet de changer
  de fournisseur d'embeddings sans modifier le pipeline.
- **Persistance disque** (`chroma_dir`) : on évite de recalculer les
  embeddings à chaque exécution, ce qui est crucial vu le coût et la
  latence de l'API UTC.

### 2.4. Cœur du RAG — `rag_pi/rag.py`

`answer_question(question, settings)` est le point d'entrée. Il enchaîne
récupération, reranking, filtrage et génération.

```
                       answer_question(q)
                              │
                              ▼
         ┌──────────────────────────────────────────┐
         │ 1. load_vectorstore(settings)            │
         │ 2. _retrieve_documents(q, k=RETRIEVER_K) │
         │      similarity_search_with_score        │
         │      ↳ fallback as_retriever() si erreur │
         └──────────────────┬───────────────────────┘
                            ▼
         ┌──────────────────────────────────────────┐
         │ 3. _rerank_documents_by_query            │
         │    score += overlap_lexical * weight     │
         │    tri décroissant                       │
         └──────────────────┬───────────────────────┘
                            ▼
         ┌──────────────────────────────────────────┐
         │ 4. _filter_documents_by_score            │
         │    élimine docs < MIN_RELEVANCE_SCORE    │
         └──────────────────┬───────────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │ pas de doc → RagAnswer de   │
              │ refus (anti-hallucination)  │
              └──────────────┬──────────────┘
                             ▼ sinon
         ┌──────────────────────────────────────────┐
         │ 5. chain = RAG_PROMPT | LLM | StrOutput  │
         │    LLM temperature=0                     │
         └──────────────────┬───────────────────────┘
                            ▼
              RagAnswer(answer, sources, context_docs, doc_scores)
```

**Leviers de performance :**

- **Récupération avec scores natifs Chroma** (`similarity_search_with_score`)
  plutôt qu'un simple `retriever.invoke()` : on conserve une métrique
  objective qui peut être affichée à l'utilisateur (UI Streamlit) et
  exploitée par le filtre/rerank.
- **Fallback gracieux** : si l'API du store ne permet pas de retourner les
  scores, on retombe sur `as_retriever().invoke()` avec un score par
  défaut — la pipeline ne casse jamais.
- **Reranking lexical hybride** :
  `_rerank_documents_by_query` calcule `overlap = |termes_q ∩ texte_doc| /
  |termes_q|` (avec `len(term) > 3` pour ignorer les mots vides) sur
  `page_content + metadata` puis ajoute `overlap * rerank_overlap_weight`
  au score vectoriel. Cela corrige les cas où les embeddings sont
  ambivalents mais où des mots-clés juridiques précis (« patrimoniaux »,
  « moraux », « L112-4 ») apparaissent littéralement. L'index utilise donc
  une **combinaison dense (embeddings) + sparse (lexical)**.
- **Pondération sur les métadonnées** : l'overlap est calculé aussi sur
  les métadonnées (titre d'article, ID), pas seulement le `page_content`.
  Une question qui mentionne explicitement « L121-1 » fait remonter le bon
  article même si son contenu n'utilise pas ce libellé.
- **Filtrage par score minimum** (`MIN_RELEVANCE_SCORE`) : les passages
  trop faibles ne sont jamais envoyés au LLM — moins de bruit dans le
  prompt, moins d'hallucinations dérivées.
- **Refus explicite quand la base ne sait pas** : si aucun document ne
  passe le filtre, on renvoie un `RagAnswer` qui dit « je ne sais pas avec
  les sources disponibles » **sans appeler le LLM**. C'est la première
  ligne de défense anti-hallucination, et c'est gratuit (pas d'appel).
- **Température 0** sur le LLM (`build_llm`) : réponses déterministes,
  reproductibles dans l'évaluation.
- **Format des documents injectés** (`format_docs`) : chaque passage est
  préfixé par `[Source N]`, son titre et son URL, ce qui aide le LLM à
  citer correctement et à attacher chaque affirmation à sa source.
- **Sources uniques** (`format_sources` avec `seen`) : on ne montre pas
  trois fois la même URL si plusieurs chunks viennent du même article.

### 2.5. Prompt système — `rag_pi/prompts.py`

Le `RAG_PROMPT` est un `ChatPromptTemplate` avec un message système strict
puis la question utilisateur. Le système :
- pose le rôle (« assistant RAG propriété intellectuelle »),
- interdit explicitement la fabrication de jurisprudence, dates,
  sanctions,
- impose la réponse **uniquement** sur le contexte,
- exige une section « Sources utilisées » en fin de réponse,
- impose un refus explicite si le contexte est insuffisant.

**Leviers de performance :**

- **Garde-fous formulés comme des règles obligatoires** : les LLM
  juridiques sont particulièrement enclins à inventer des numéros
  d'articles ; le prompt nomme les pièges les plus fréquents
  (jurisprudence, dates, sanctions).
- **Format de sortie contraint** (section « Sources utilisées ») : facilite
  l'évaluation automatique (regex/term matching) et la lecture humaine.
- **Refus explicite encouragé** : couplé au refus de §2.4, on obtient
  deux niveaux d'abstention (silencieux si retrieve vide ; explicite si
  retrieve faible).

### 2.6. Évaluation automatique — `rag_pi/evaluation.py`, `evaluation_cases.py`, `scripts/evaluate_rag.py`

L'évaluation s'appuie sur des `EvalCase` typés :

```python
EvalCase(
    id, question,
    expected_terms,        # mots-clés qui doivent apparaître dans la réponse
    expected_sources,      # fragments d'URL/titre qui doivent être cités
    expected_answer_snippet,
    category,              # single-hop / multi-hop / hallucination
    should_refuse,         # True pour les cas piège
)
```

Pour chaque cas, `evaluate()` appelle `answer_question`, puis calcule :
- `matched_terms` / `term_match_ratio`,
- `matched_sources` / `source_match_ratio`,
- `snippet_match` (présence d'un extrait clé),
- `grounded` (au moins une source citée),
- `passed` (logique différenciée : un cas `should_refuse` est validé si
  le système répond par un refus détecté via `_detect_refusal`).

`scripts/evaluate_rag.py` exporte les résultats dans
`evaluation/results/latest.{json,csv}` et imprime un score global.

```
┌────────────────────────────┐
│  evaluation_cases.py       │
│  DEFAULT_EVAL_CASES        │
│   - single-hop             │
│   - multi-hop              │
│   - hallucination (refus)  │
└──────────────┬─────────────┘
               │
               ▼
   evaluate(settings, cases)
               │
               │   pour chaque cas
               ▼
   answer_question(q) ──► RagAnswer
               │
               ▼
   match_terms / match_sources / match_snippet / detect_refusal
               │
               ▼
   EvalResult (term_match_ratio, source_match_ratio, grounded, passed)
               │
               ▼
   latest.json  /  latest.csv
```

**Leviers de performance :**

- **Cas typés par catégorie** (`single-hop`, `multi-hop`,
  `hallucination`) : permet d'identifier précisément si une régression
  vient du retrieve, de la composition multi-articles ou du contrôle
  d'hallucination.
- **Cas explicitement piégés** (`hallucination_brevets_quantiques`) : on
  teste que le système refuse de fabriquer une jurisprudence inexistante,
  ce qui valide le couple « prompt strict + filtre min_score ».
- **Détection de refus tolérante** (`_detect_refusal`) : couvre plusieurs
  formulations (« ne sais pas », « aucune source », « pas de source »…),
  pour éviter qu'une variation stylistique du LLM ne casse la mesure.
- **Métriques continues plutôt que binaires** : `term_match_ratio` et
  `source_match_ratio` permettent de suivre une progression douce du
  système entre deux runs même quand `passed` reste identique.
- **Export JSON + CSV** : `latest.json` pour la lecture détaillée (toutes
  les sources, l'answer complet), `latest.csv` pour l'analyse rapide en
  tableur. Un seul fichier `latest.*` simplifie le pilotage.
- **Reproductibilité** : `temperature=0` côté LLM rend les écarts entre
  deux runs imputables aux changements de configuration, pas au hasard de
  l'échantillonnage.

### 2.7. Interface Streamlit — `streamlit_app.py`

L'UI affiche un chat, appelle `answer_question` pour chaque question,
montre la réponse, la liste des sources avec leur **score**, et un
expander qui détaille chaque document récupéré (titre, score, URL,
extrait de 600 caractères).

**Leviers de performance :**

- **`@st.cache_resource` sur `load_settings`** : la configuration n'est
  lue qu'une fois par session ; évite des relectures `.env` inutiles.
- **Affichage des scores** : la transparence sur la qualité du retrieve
  permet à l'utilisateur de juger lui-même si la réponse est fiable et
  facilite le debug pendant les démos.
- **Détails dépliables** (`expander`) : on garde l'écran propre tout en
  permettant la vérification fine — utile pour démontrer le grounding en
  soutenance sans noyer l'utilisateur final.
- **Gestion d'erreur de configuration** (`RuntimeError` → `st.error` +
  `st.stop`) : si `.env` est mal renseigné, l'app affiche un message
  clair au lieu de planter en plein milieu d'une requête.

---

## 3. Récapitulatif des mesures anti-hallucination et de performance

Les mesures se renforcent les unes les autres. Le tableau ci-dessous les
résume :

| Catégorie                | Mesure                                                        | Module              |
|--------------------------|---------------------------------------------------------------|---------------------|
| Qualité du corpus        | Crawl borné, déduplication par `article_id` et par contenu    | `loaders.py`        |
| Qualité du corpus        | Métadonnées riches (`article_id`, `article_title`, URL ancrée)| `loaders.py`        |
| Découpage                | Chunking récursif avec chevauchement (1200 / 180)             | `ingest_legifrance` |
| Robustesse ingestion     | Batchs + sleep + retry exponentiel sur embeddings             | `vectorstore.py`    |
| Persistance              | Chroma local persistant + IDs stables                         | `vectorstore.py`    |
| Rappel                   | `similarity_search_with_score` (top-k configurable)           | `rag.py`            |
| Précision (hybride)      | Reranking lexical par overlap mots-clés (dense + sparse)      | `rag.py`            |
| Précision                | Filtrage par `MIN_RELEVANCE_SCORE`                            | `rag.py`            |
| Anti-hallucination       | Refus court-circuit si aucun doc retenu                       | `rag.py`            |
| Anti-hallucination       | Prompt système strict (pas d'invention, sources obligatoires) | `prompts.py`        |
| Déterminisme             | `temperature=0` sur le LLM                                    | `rag.py`            |
| Traçabilité              | Sources + scores affichés dans l'UI                           | `streamlit_app.py`  |
| Pilotage qualité         | Cas d'évaluation typés + détection de refus                   | `evaluation*.py`    |
| Pilotage qualité         | Export JSON/CSV pour suivi de régression                      | `evaluate_rag.py`   |

---

## 4. Pour aller plus loin

Pistes d'amélioration cohérentes avec l'architecture actuelle :

- **Reranker neuronal** (`cross-encoder` type `ms-marco-MiniLM`) en plus
  du reranking lexical actuel — meilleure précision sur les multi-hop.
- **Chunking par article** plutôt que par taille fixe : un article du
  Code = une unité naturelle, ce qui supprime les coupures en plein
  milieu d'une définition.
- **Cache de réponses** par question normalisée pour éviter de
  re-solliciter le LLM en démo.
- **Élargir le jeu d'évaluation** (plus de multi-hop, cas adverses
  formulés en langage naturel ambigu) et tracker l'évolution dans le
  temps plutôt qu'un unique `latest.*`.
