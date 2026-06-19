"""Evidence-quality metrics and confidence scoring (no ensemble — single model)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "was",
    "were", "by", "with", "as", "at", "be", "this", "that", "his", "her", "its", "their",
    "from", "into", "during", "known", "best", "what", "which", "who", "primarily",
}


def evidence_quality(items: List[Dict[str, str]]) -> Dict:
    """Summarise retrieved evidence for logging + scoring."""
    n_docs = len(items)
    n_wiki = sum(1 for it in items if it.get("source") == "wikipedia")
    total_chars = sum(len(it.get("snippet", "")) for it in items)
    return {
        "n_docs": n_docs,
        "n_wikipedia": n_wiki,
        "total_chars": total_chars,
        "has_wikipedia": n_wiki > 0,
    }


def _significant_words(text: str) -> List[str]:
    return [w for w in (m.group(0).lower() for m in _WORD.finditer(text or ""))
            if len(w) > 3 and w not in _STOP]


def support_score(letter: Optional[str], options: Dict[str, str], evidence_text: str) -> float:
    """Fraction of the chosen option's significant words that appear in the evidence."""
    if not letter or letter not in options:
        return 0.0
    words = _significant_words(options[letter])
    if not words:
        return 0.0
    ev = (evidence_text or "").lower()
    hits = sum(1 for w in words if w in ev)
    return round(hits / len(words), 3)


def _parse_score(parsed_from: str) -> float:
    return {"pattern": 1.0, "option_text": 0.7, "fallback": 0.0}.get(parsed_from, 0.0)


def _evidence_score(eq: Dict) -> float:
    base = min(1.0, eq["n_docs"] / 5.0)
    if eq["has_wikipedia"]:
        base = 0.4 + 0.6 * base  # floor of 0.4 once Wikipedia evidence is present
    return round(base, 3)


def compute_confidence(
    primary: Dict,
    rag: Optional[Dict],
    cb: Optional[Dict],
    items: List[Dict[str, str]],
    evidence_text: str,
    options: Dict[str, str],
) -> Dict:
    """Combine parse success, evidence quality, evidence support, and RAG/closed-book
    agreement into a single confidence in [0, 1] plus its components."""
    eq = evidence_quality(items)
    parse = _parse_score(primary.get("parsed_from", "fallback"))
    evid = _evidence_score(eq)
    support = support_score(primary.get("answer"), options, evidence_text)

    agreement: Optional[float] = None
    if rag is not None and cb is not None:
        agreement = 1.0 if rag["answer"] == cb["answer"] else 0.0

    if agreement is not None:
        conf = 0.20 * parse + 0.20 * evid + 0.25 * support + 0.35 * agreement
    else:
        conf = 0.30 * parse + 0.30 * evid + 0.40 * support

    return {
        "confidence": round(max(0.0, min(1.0, conf)), 3),
        "parse_score": parse,
        "evidence_score": evid,
        "support_score": support,
        "agreement": agreement,
        "n_evidence": eq["n_docs"],
        "has_wikipedia": eq["has_wikipedia"],
    }


def top_evidence(items: List[Dict[str, str]], k: int = 3, snippet_chars: int = 240) -> List[Dict]:
    out = []
    for it in items[:k]:
        out.append({
            "title": it.get("title", ""),
            "source": it.get("source", ""),
            "snippet": (it.get("snippet", "") or "")[:snippet_chars],
        })
    return out
