from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

from rag_pi.config import Settings
from rag_pi.prompts import RAG_PROMPT
from rag_pi.vectorstore import load_vectorstore

# Ce module gère le cœur du workflow RAG : récupération, reranking et génération.
# Les fonctions sont conçues pour rester simples et explicables.


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[dict[str, str]]
    context_docs: list[Document]
    doc_scores: list[float]


# Construction simple d'un client LLM pour la génération de texte.
# On utilise un modèle compatible avec l'API UTC et on fixe la température à 0 pour obtenir des réponses plus déterministes.
def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
    )


def answer_question(question: str, settings: Settings) -> RagAnswer:
    # 1) Charger le vectorstore Chroma
    # 2) Récupérer les documents les plus proches
    # 3) Reranker selon le recouvrement lexical
    # 4) Filtrer les documents trop faibles
    vectorstore = load_vectorstore(settings)
    docs_with_scores = _retrieve_documents(vectorstore, question, settings.retriever_k)
    docs_with_scores = _rerank_documents_by_query(docs_with_scores, question, settings.rerank_overlap_weight)
    docs_with_scores = _filter_documents_by_score(docs_with_scores, settings.min_relevance_score)
    docs = [doc for doc, _ in docs_with_scores]
    scores = [score for _, score in docs_with_scores]

    if not docs:
        return RagAnswer(
            answer=(
                "Je ne sais pas avec les sources disponibles. "
                "La base documentaire Chroma ne contient aucun passage pertinent."
            ),
            sources=[],
            context_docs=[],
            doc_scores=[],
        )

    chain = RAG_PROMPT | build_llm(settings) | StrOutputParser()
    answer = chain.invoke({"question": question, "context": format_docs(docs)})
    return RagAnswer(
        answer=answer,
        sources=format_sources(docs),
        context_docs=docs,
        doc_scores=scores,
    )


def _retrieve_documents(
    vectorstore: Chroma, question: str, k: int
) -> list[tuple[Document, float]]:
    # Tente d'abord d'obtenir des documents avec score depuis Chroma.
    # Si l'API du vectorstore ne le permet pas, on récupère les docs sans score.
    if hasattr(vectorstore, "similarity_search_with_score"):
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=k)
            if docs_with_scores:
                return docs_with_scores
        except Exception:
            pass

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return [(doc, 1.0) for doc in docs]


def _normalize_text(text: str) -> str:
    # Nettoie le texte pour comparer facilement les mots.
    return " ".join(text.lower().split())


def _query_overlap_score(question: str, doc: Document) -> float:
    # Calcule le pourcentage de mots importants de la question qui apparaissent dans le document ou ses métadonnées.
    question_terms = set(term for term in _normalize_text(question).split() if len(term) > 3)
    content = _normalize_text(doc.page_content)
    metadata_text = _normalize_text(" ".join(str(value) for value in doc.metadata.values()))
    text = f"{content} {metadata_text}"
    overlap = sum(1 for term in question_terms if term in text)
    return overlap / max(1, len(question_terms))


def _rerank_documents_by_query(
    docs_with_scores: list[tuple[Document, float]], question: str, weight: float
) -> list[tuple[Document, float]]:
    # Si on a un poids de reranking, on ajoute un bonus au score.
    # Cela aide à favoriser les documents qui partagent des mots importants avec la question.
    if weight <= 0.0:
        return docs_with_scores

    reranked: list[tuple[Document, float]] = []
    for doc, score in docs_with_scores:
        bonus = _query_overlap_score(question, doc) * weight
        reranked.append((doc, score + bonus))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def _filter_documents_by_score(
    docs_with_scores: list[tuple[Document, float]],
    min_score: float,
) -> list[tuple[Document, float]]:
    # Élimine les documents dont le score est trop faible.
    if min_score <= 0.0:
        return docs_with_scores
    return [(doc, score) for doc, score in docs_with_scores if score >= min_score]


def format_docs(docs: list[Document]) -> str:
    # Prépare le texte envoyé au modèle en concaténant les documents avec leurs titres et URL de source.
    chunks = []
    for index, doc in enumerate(docs, start=1):
        title = doc.metadata.get("article_title") or doc.metadata.get("title") or "Source"
        source = doc.metadata.get("source", "")
        chunks.append(f"[Source {index}] {title}\nURL: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


def format_sources(docs: list[Document]) -> list[dict[str, str]]:
    # Récupère une liste de sources unique pour l'affichage final.
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source in seen:
            continue
        sources.append(
            {
                "title": doc.metadata.get("article_title") or doc.metadata.get("title") or "Source",
                "url": source,
            }
        )
        seen.add(source)
    return sources
