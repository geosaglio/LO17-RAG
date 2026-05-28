from __future__ import annotations

from urllib.parse import urljoin, urlparse

import requests

from rag_pi.config import load_settings


def main() -> None:
    settings = load_settings()
    text = "Test court pour verifier le service d'embeddings."
    headers = {
        "Authorization": f"Bearer {settings.embedding_api_key}",
        "Content-Type": "application/json",
    }
    parsed_base = urlparse(settings.embedding_base_url)
    root_base_url = f"{parsed_base.scheme}://{parsed_base.netloc}"

    candidates = [
        (
            "openai:/embeddings input",
            settings.embedding_base_url,
            "/embeddings",
            {"model": settings.embedding_model, "input": text, "encoding_format": "float"},
        ),
        (
            "openai:/embeddings input-array",
            settings.embedding_base_url,
            "/embeddings",
            {"model": settings.embedding_model, "input": [text], "encoding_format": "float"},
        ),
        (
            "openwebui:/embeddings prompt",
            settings.embedding_base_url,
            "/embeddings",
            {"model": settings.embedding_model, "prompt": text},
        ),
        (
            "openwebui:/embeddings input",
            settings.embedding_base_url,
            "/embeddings",
            {"model": settings.embedding_model, "input": text},
        ),
        (
            "ollama-root:/ollama/api/embeddings prompt",
            root_base_url,
            "/ollama/api/embeddings",
            {"model": settings.embedding_model, "prompt": text},
        ),
        (
            "ollama-root:/ollama/api/embed input",
            root_base_url,
            "/ollama/api/embed",
            {"model": settings.embedding_model, "input": text},
        ),
        (
            "ollama-api-base:/ollama/api/embeddings prompt",
            settings.embedding_base_url,
            "/ollama/api/embeddings",
            {"model": settings.embedding_model, "prompt": text},
        ),
        (
            "ollama-api-base:/ollama/api/embed input",
            settings.embedding_base_url,
            "/ollama/api/embed",
            {"model": settings.embedding_model, "input": text},
        ),
        (
            "v1:/v1/embeddings input",
            settings.embedding_base_url,
            "/v1/embeddings",
            {"model": settings.embedding_model, "input": text, "encoding_format": "float"},
        ),
    ]

    for label, base_url, path, payload in candidates:
        url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
        print(f"\n== {label}")
        print(f"POST {url}")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            print(f"REQUEST ERROR: {exc}")
            continue

        print(f"HTTP {response.status_code}")
        print(response.text[:1000])

        if response.ok:
            vector = _extract_embedding(response.json())
            if vector:
                print(f"Embedding OK: dimension={len(vector)}")
                return

    raise SystemExit("Aucun format d'embedding teste n'a fonctionne.")


def _extract_embedding(payload):
    if "embedding" in payload:
        return payload["embedding"]
    if "embeddings" in payload and payload["embeddings"]:
        return payload["embeddings"][0]
    data = payload.get("data")
    if data and "embedding" in data[0]:
        return data[0]["embedding"]
    return None


if __name__ == "__main__":
    main()
