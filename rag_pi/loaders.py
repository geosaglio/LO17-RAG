from __future__ import annotations

from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


DEFAULT_HEADERS = {
    "User-Agent": "lo17-rag-pi/0.1 (+https://www.utc.fr; educational project)"
}


def load_legifrance_code(url: str) -> list[Document]:
    """Load the public Legifrance code page and split it by visible article blocks."""
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    page_title = soup.title.get_text(" ", strip=True) if soup.title else "Code de la propriete intellectuelle"

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    article_nodes = soup.select("article.js-article-content-to-copy")
    docs = _documents_from_articles(article_nodes, url, page_title)
    if docs:
        return docs

    main = soup.select_one("main") or soup.body or soup
    text = _clean_text(main.get_text("\n", strip=True))
    return [
        Document(
            page_content=text,
            metadata={
                "source": url,
                "title": page_title,
                "article_id": "",
                "article_title": page_title,
            },
        )
    ]


def crawl_legifrance_code(start_url: str, max_pages: int = 25) -> list[Document]:
    """Crawl public Legifrance code pages reachable from the entry URL."""
    queue = deque([start_url])
    visited: set[str] = set()
    seen_article_ids: set[str] = set()
    documents: list[Document] = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue

        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        visited.add(url)

        soup = BeautifulSoup(response.text, "html.parser")
        page_title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else "Code de la propriete intellectuelle"
        )

        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()

        article_nodes = soup.select("article.js-article-content-to-copy")
        docs = _documents_from_articles(article_nodes, url, page_title)
        if docs:
            for doc in docs:
                article_id = doc.metadata.get("article_id", "")
                if article_id and article_id in seen_article_ids:
                    continue
                if article_id:
                    seen_article_ids.add(article_id)
                documents.append(doc)
        elif _should_keep_fallback_document(url, start_url):
            main = soup.select_one("main") or soup.body or soup
            text = _clean_text(main.get_text("\n", strip=True))
            if len(text) > 200:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "title": page_title,
                            "article_id": "",
                            "article_title": page_title,
                        },
                    )
                )

        for link in _legifrance_links(soup, url, start_url):
            if link not in visited and link not in queue:
                queue.append(link)

    return documents


def _documents_from_articles(nodes, base_url: str, page_title: str) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()

    for node in nodes:
        heading_node = node.select_one(".name-article, .article_title, h2")
        content_node = node.select_one(".content")
        date_node = node.select_one(".date")
        badge_node = node.select_one(".fr-badge--success, .lgf-dsfr-badge")

        heading = (
            heading_node.get_text(" ", strip=True)
            if heading_node
            else _nearest_heading(node) or _first_line(node.get_text("\n", strip=True))
        )
        article_id = (
            heading_node.get("data-anchor", "")
            if heading_node
            else ""
        ) or _article_id_from_link(node)

        parts = [heading]
        if badge_node:
            parts.append(badge_node.get_text(" ", strip=True))
        if date_node:
            parts.append(date_node.get_text(" ", strip=True))
        if content_node:
            parts.append(content_node.get_text("\n", strip=True))

        text = _clean_text("\n\n".join(parts))
        if len(text) < 80 or text in seen:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": urljoin(base_url, f"#{article_id}") if article_id else base_url,
                    "title": page_title,
                    "article_id": article_id,
                    "article_title": heading,
                },
            )
        )
        seen.add(text)

    return docs


def _article_id_from_link(node) -> str:
    link = node.select_one("a[href*='LEGIARTI']")
    if not link:
        return ""
    href = link.get("href", "")
    parsed = urlparse(href)
    return parsed.path.rsplit("/", 1)[-1] if "LEGIARTI" in parsed.path else ""


def _nearest_heading(node) -> str:
    heading = node.find(["h1", "h2", "h3", "h4"])
    if heading:
        return heading.get_text(" ", strip=True)

    previous = node.find_previous(["h1", "h2", "h3", "h4"])
    return previous.get_text(" ", strip=True) if previous else ""


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _legifrance_links(soup: BeautifulSoup, current_url: str, start_url: str) -> list[str]:
    start_path = urlparse(start_url).path
    links: list[str] = []

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        absolute = urljoin(current_url, href).split("#", 1)[0]
        parsed = urlparse(absolute)
        if parsed.netloc != "www.legifrance.gouv.fr":
            continue
        if not parsed.path.startswith("/codes/"):
            continue
        if "LEGITEXT000006069414" not in parsed.path and parsed.path != start_path:
            continue
        links.append(absolute)

    return links


def _should_keep_fallback_document(url: str, start_url: str) -> bool:
    parsed = urlparse(url)
    start_path = urlparse(start_url).path
    if parsed.path == start_path or parsed.path.startswith(start_path + "/"):
        return False
    if "/codes/texte_lc/" in parsed.path:
        return False
    return "/codes/section_lc/" not in parsed.path
