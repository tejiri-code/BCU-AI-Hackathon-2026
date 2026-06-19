"""Best-effort web search via DuckDuckGo (ddgs). Tolerates rate limits."""
from __future__ import annotations

from typing import Dict, List

try:
    from ddgs import DDGS
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


def retrieve(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    if not _AVAILABLE:
        return []
    out: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "source": item.get("href", "web"),
                })
    except Exception:  # network / rate-limit: degrade gracefully
        return out
    return out
