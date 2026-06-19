"""Assemble cached evidence for a question: Wikipedia (primary) + web (fallback)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from . import query as query_mod
from . import web, wikipedia


def _cache_key(question_no, queries: List[str]) -> str:
    h = hashlib.sha1(("||".join(queries)).encode("utf-8")).hexdigest()[:12]
    return f"q{question_no}_{h}"


def gather_evidence(
    question_no,
    question: str,
    options: Dict[str, str],
    cfg: Dict,
    cache_dir: Path,
) -> Dict:
    """Return {queries, items:[{title,snippet,source}]}, cached to disk by question."""
    queries = query_mod.build_queries(question, options)
    cache_dir = Path(cache_dir) / "retrieval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(question_no, queries)}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    items: List[Dict[str, str]] = []
    seen_titles = set()

    # Wikipedia first (questions are phrased "according to Wikipedia").
    for q in queries:
        for it in wikipedia.retrieve(q, cfg["wikipedia_articles"], cfg["wikipedia_chars"]):
            key = it["title"].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                items.append(it)

    # Web as supplementary evidence.
    if cfg.get("enable_web", True):
        for it in web.retrieve(queries[0], cfg["web_results"]):
            items.append(it)

    result = {"question_no": question_no, "queries": queries, "items": items}
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def format_evidence(items: List[Dict[str, str]], char_budget: int = 4000) -> str:
    """Concatenate evidence snippets into a prompt block within a char budget."""
    blocks, used = [], 0
    for it in items:
        title = it.get("title", "").strip()
        snippet = it.get("snippet", "").strip()
        if not snippet:
            continue
        piece = f"[{title}] {snippet}" if title else snippet
        if used + len(piece) > char_budget:
            piece = piece[: max(0, char_budget - used)]
        if piece:
            blocks.append(piece)
            used += len(piece)
        if used >= char_budget:
            break
    return "\n\n".join(blocks)
