"""Robust extraction of a single MCQ answer letter (A-E) from model text."""
from __future__ import annotations

import re
from typing import Dict, Optional

from . import OPTION_LETTERS

VALID = set(OPTION_LETTERS)

# Ordered patterns, most explicit first.
_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=|->)?\s*\(?([A-E])\)?", re.IGNORECASE),
    re.compile(r"\boption\s*\(?([A-E])\)?", re.IGNORECASE),
    re.compile(r"\bcorrect\s+(?:answer|option)\s+is\s*\(?([A-E])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-E])\)?[\.\):]", re.IGNORECASE | re.MULTILINE),
]


def parse_letter(text: str) -> Optional[str]:
    """Return a single uppercase letter A-E, or None if nothing confidently matches."""
    if not text:
        return None
    cleaned = text.strip()

    # Prefer the last explicit "Answer: X" style match (models often restate at the end).
    for pat in _PATTERNS:
        matches = pat.findall(cleaned)
        if matches:
            cand = matches[-1].upper()
            if cand in VALID:
                return cand

    # Fallback: a lone letter on its own line.
    for line in reversed(cleaned.splitlines()):
        token = line.strip().strip(".)( :*").upper()
        if token in VALID:
            return token

    # Last resort: the final standalone A-E token anywhere.
    loose = re.findall(r"\b([A-E])\b", cleaned)
    if loose:
        return loose[-1].upper()
    return None


def match_by_option_text(text: str, options: Dict[str, str]) -> Optional[str]:
    """If the model echoed an option's full text instead of a letter, map it back."""
    if not text:
        return None
    low = text.lower()
    best, best_len = None, 0
    for letter, opt in options.items():
        opt = (opt or "").strip().lower()
        if opt and opt in low and len(opt) > best_len:
            best, best_len = letter, len(opt)
    return best
