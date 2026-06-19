"""Wikipedia retrieval via the public MediaWiki API (no API key needed)."""
from __future__ import annotations

from typing import Dict, List

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ApexMind-BCU-Hackathon/0.1 (educational; contact via GitHub)"}


def _get(params: Dict) -> Dict:
    params = {"format": "json", **params}
    r = requests.get(API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def search_titles(query: str, limit: int = 3) -> List[str]:
    data = _get({"action": "query", "list": "search", "srsearch": query, "srlimit": limit})
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def page_extract(title: str, chars: int = 2400) -> str:
    """Plain-text intro/extract for a page, truncated to `chars`."""
    data = _get({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "titles": title,
        "exchars": chars,
    })
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "")
        if text:
            return text.strip()
    return ""


def retrieve(query: str, n_articles: int = 3, chars: int = 2400) -> List[Dict[str, str]]:
    """Return [{title, snippet, source}] for the top Wikipedia articles for `query`."""
    out: List[Dict[str, str]] = []
    try:
        titles = search_titles(query, limit=n_articles)
    except requests.RequestException:
        return out
    for title in titles:
        try:
            text = page_extract(title, chars=chars)
        except requests.RequestException:
            text = ""
        if text:
            out.append({"title": title, "snippet": text, "source": "wikipedia"})
    return out
