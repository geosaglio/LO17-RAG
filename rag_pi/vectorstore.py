from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin

import requests
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from rag_pi.config import Settings

# Stockage des embeddings et des vecteurs.
# Ce module rend possible la conversion de texte en embeddings et la création du store Chroma pour la recherche vectorielle.


class SingleTextOpenAICompatibleEmbeddings(Embeddings):
    """Client d'embeddings simple pour serveurs OpenAI-compatible.

    Certains serveurs ne gèrent pas bien les appels groupés. Cette classe
    sait faire un appel à la fois si nécessaire.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        path: str = "/embeddings",
        api_format: str = "openwebui",
    ) -> None:
        self.url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
        self.api_key = api_key
        self.model = model
        self.api_format = api_format

    def embed_query(self, text: str) -> list[float]:
        # Embeddings pour une requête unique.
        return self._embed_one(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Embeddings pour plusieurs textes. On prend en charge plusieurs formats.
        if self.api_format == "ollama_embed":
            vector = self._post({"model": self.model, "input": texts})
            if isinstance(vector[0], list):
                return vector
            return [vector]
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        payload = self._payload(text)
        return self._post(payload)

    def _post(self, payload: dict[str, Any]) -> list[float] | list[list[float]]:
        # Envoie la requête HTTP au service d'embeddings.
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Embedding API error {response.status_code} on {self.url}: {response.text[:1000]}"
            )

        payload: dict[str, Any] = response.json()
        if "embedding" in payload:
            return payload["embedding"]
        if "embeddings" in payload and payload["embeddings"]:
            return payload["embeddings"]

        data = payload.get("data")
        if not data or "embedding" not in data[0]:
            raise RuntimeError(f"Unexpected embedding API response: {payload}")
        return data[0]["embedding"]

    def embed_documents_with_retry(
        self,
        texts: list[str],
        max_retries: int = 5,
        base_sleep_seconds: float = 1.0,
    ) -> list[list[float]]:
        # Réessaie l'appel en cas d'erreur réseau temporaire.
        for attempt in range(max_retries + 1):
            try:
                return self.embed_documents(texts)
            except requests.RequestException:
                if attempt >= max_retries:
                    raise
                time.sleep(base_sleep_seconds * (2**attempt))

    def _payload(self, text: str) -> dict[str, Any]:
        if self.api_format == "openai":
            return {
                "model": self.model,
                "input": text,
                "encoding_format": "float",
            }
        if self.api_format == "openwebui":
            return {
                "model": self.model,
                "prompt": text,
            }
        if self.api_format == "ollama_embed":
            return {
                "model": self.model,
                "input": text,
            }
        if self.api_format == "ollama_embeddings":
            return {
                "model": self.model,
                "prompt": text,
            }
        raise ValueError(f"Format d'API embeddings non supporte: {self.api_format}")


def build_embeddings(settings: Settings) -> SingleTextOpenAICompatibleEmbeddings:
    # Crée un client d'embeddings à partir des paramètres du projet.
    return SingleTextOpenAICompatibleEmbeddings(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        path=settings.embedding_path,
        api_format=settings.embedding_api_format,
    )


def load_vectorstore(settings: Settings) -> Chroma:
    # Initialise le dossier Chroma et retourne le store prêt à l'emploi.
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=build_embeddings(settings),
    )
