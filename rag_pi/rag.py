from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag_pi.config import Settings
from rag_pi.prompts import RAG_PROMPT
from rag_pi.vectorstore import load_vectorstore


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[dict[str, str]]
    context_docs: list[Document]


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
    )


def answer_question(question: str, settings: Settings) -> RagAnswer:
    vectorstore = load_vectorstore(settings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})
    docs = retriever.invoke(question)

    if not docs:
        return RagAnswer(
            answer=(
                "Je ne sais pas avec les sources disponibles. "
                "La base documentaire Chroma ne contient aucun passage pertinent."
            ),
            sources=[],
            context_docs=[],
        )

    chain = RAG_PROMPT | build_llm(settings) | StrOutputParser()
    answer = chain.invoke({"question": question, "context": format_docs(docs)})
    return RagAnswer(answer=answer, sources=format_sources(docs), context_docs=docs)


def format_docs(docs: list[Document]) -> str:
    chunks = []
    for index, doc in enumerate(docs, start=1):
        title = doc.metadata.get("article_title") or doc.metadata.get("title") or "Source"
        source = doc.metadata.get("source", "")
        chunks.append(f"[Source {index}] {title}\nURL: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


def format_sources(docs: list[Document]) -> list[dict[str, str]]:
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
