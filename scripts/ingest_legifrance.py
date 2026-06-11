from __future__ import annotations

import argparse
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pi.config import load_settings
from rag_pi.loaders import crawl_legifrance_code
from rag_pi.vectorstore import build_embeddings, load_vectorstore

# Script d'ingestion chargé de récupérer le texte Legifrance,
# de le découper en chunks, de calculer les embeddings et de les
# stocker dans Chroma.


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion Legifrance vers Chroma.")
    parser.add_argument("--reset", action="store_true", help="Recree la collection Chroma.")
    args = parser.parse_args()

    settings = load_settings()
    docs = crawl_legifrance_code(settings.legifrance_url, max_pages=settings.max_pages)

    # Sépare les longs textes en morceaux plus petits pour Chroma.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vectorstore = load_vectorstore(settings)
    if args.reset:
        vectorstore.delete_collection()
        vectorstore = load_vectorstore(settings)
    embeddings = vectorstore._embedding_function or build_embeddings(settings)

    # Ajoute les vecteurs et docs à la collection Chroma par lots.

    ids = [_stable_id(chunk, index) for index, chunk in enumerate(chunks)]
    for start in range(0, len(chunks), settings.embedding_batch_size):
        end = min(start + settings.embedding_batch_size, len(chunks))
        batch_chunks = chunks[start:end]
        batch_ids = ids[start:end]
        first_title = batch_chunks[0].metadata.get("article_title", "")
        last_title = batch_chunks[-1].metadata.get("article_title", "")
        print(
            f"Embedding chunks {start + 1}-{end}/{len(chunks)} | "
            f"{first_title} -> {last_title}"
        )

        texts = [chunk.page_content for chunk in batch_chunks]
        if hasattr(embeddings, "embed_documents_with_retry"):
            vectors = embeddings.embed_documents_with_retry(
                texts,
                max_retries=settings.embedding_max_retries,
                base_sleep_seconds=settings.embedding_sleep_seconds,
            )
        else:
            vectors = embeddings.embed_documents(texts)

        vectorstore._collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=[chunk.metadata for chunk in batch_chunks],
        )
        time.sleep(settings.embedding_sleep_seconds)
    vectorstore.persist()

    print(f"Ingestion terminee: {len(docs)} documents sources, {len(chunks)} chunks.")

def _stable_id(chunk, index: int) -> str:
    article_id = chunk.metadata.get("article_id") or "page"
    return f"{article_id}-{index}"


if __name__ == "__main__":
    main()
