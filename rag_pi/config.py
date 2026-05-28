from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    embedding_base_url: str
    embedding_api_key: str
    llm_model: str = "mistral-small3.2:latest"
    embedding_model: str = "rhundt/GLM-4-0414-32b-128k-Q4_K_M:latest"
    embedding_path: str = "/embeddings"
    embedding_api_format: str = "openwebui"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "legifrance_propriete_intellectuelle"
    legifrance_url: str = (
        "https://www.legifrance.gouv.fr/codes/texte_lc/LEGITEXT000006069414"
    )
    chunk_size: int = 1200
    chunk_overlap: int = 180
    retriever_k: int = 5
    max_pages: int = 25
    embedding_batch_size: int = 8
    embedding_sleep_seconds: float = 1.0
    embedding_max_retries: int = 5


def load_settings() -> Settings:
    load_dotenv()

    llm_base_url = os.getenv("LLM_BASE_URL", "").strip()
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip()

    missing = [
        name
        for name, value in {
            "LLM_BASE_URL": llm_base_url,
            "LLM_API_KEY": llm_api_key,
            "EMBEDDING_BASE_URL": embedding_base_url,
            "EMBEDDING_API_KEY": embedding_api_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Variables manquantes dans l'environnement ou .env: {', '.join(missing)}")

    return Settings(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        embedding_path=os.getenv("EMBEDDING_PATH", Settings.embedding_path),
        embedding_api_format=os.getenv(
            "EMBEDDING_API_FORMAT", Settings.embedding_api_format
        ),
        chroma_dir=Path(os.getenv("CHROMA_DIR", Settings.chroma_dir)),
        collection_name=os.getenv("CHROMA_COLLECTION", Settings.collection_name),
        legifrance_url=os.getenv("LEGIFRANCE_URL", Settings.legifrance_url),
        chunk_size=int(os.getenv("CHUNK_SIZE", Settings.chunk_size)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", Settings.chunk_overlap)),
        retriever_k=int(os.getenv("RETRIEVER_K", Settings.retriever_k)),
        max_pages=int(os.getenv("LEGIFRANCE_MAX_PAGES", Settings.max_pages)),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", Settings.embedding_batch_size)),
        embedding_sleep_seconds=float(
            os.getenv("EMBEDDING_SLEEP_SECONDS", Settings.embedding_sleep_seconds)
        ),
        embedding_max_retries=int(
            os.getenv("EMBEDDING_MAX_RETRIES", Settings.embedding_max_retries)
        ),
    )
