from __future__ import annotations

from urllib.parse import urlparse

import requests

from rag_pi.config import load_settings


def main() -> None:
    settings = load_settings()
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    parsed_base = urlparse(settings.llm_base_url)
    root_base_url = f"{parsed_base.scheme}://{parsed_base.netloc}"

    candidates = [
        ("openai models", f"{settings.llm_base_url.rstrip('/')}/models"),
        ("openai v1 models", f"{settings.llm_base_url.rstrip('/')}/v1/models"),
        ("ollama tags", f"{root_base_url}/ollama/api/tags"),
        ("openwebui models", f"{root_base_url}/api/models"),
    ]

    for label, url in candidates:
        print(f"\n== {label}")
        print(f"GET {url}")
        try:
            response = requests.get(url, headers=headers, timeout=60)
        except requests.RequestException as exc:
            print(f"REQUEST ERROR: {exc}")
            continue

        print(f"HTTP {response.status_code}")
        print(response.text[:5000])


if __name__ == "__main__":
    main()
