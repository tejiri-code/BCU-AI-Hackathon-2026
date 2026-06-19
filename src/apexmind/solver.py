"""Answer a single MCQ with one model (deterministic pass + robust parsing)."""
from __future__ import annotations

from typing import Dict

from . import parsing, prompting, scoring
from .llm_client import LlamaClient
from .retrieval import evidence as ev

MODES = ("closed_book", "rag", "rag_confidence")


def answer_one(
    client: LlamaClient,
    question: str,
    options: Dict[str, str],
    evidence: str,
    gen: Dict,
) -> Dict:
    """Return {answer, raw, parsed_from} for a single question."""
    messages = prompting.build_messages(
        question, options, evidence, disable_thinking=gen.get("disable_thinking", True)
    )
    raw = client.chat(
        messages,
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("max_tokens", 512),
        seed=gen.get("seed"),
    )

    letter = parsing.parse_letter(raw)
    parsed_from = "pattern"
    if letter is None:
        letter = parsing.match_by_option_text(raw, options)
        parsed_from = "option_text"
    if letter is None:
        # Never waste a question: fall back to the first option rather than Unknown.
        letter = next(iter(options), "A")
        parsed_from = "fallback"

    return {"answer": letter, "raw": raw, "parsed_from": parsed_from}


def solve_question(
    client: LlamaClient,
    question: str,
    options: Dict[str, str],
    evidence_items,
    gen: Dict,
    mode: str = "rag_confidence",
    char_budget: int = 4000,
) -> Dict:
    """Answer one question in a given mode and attach confidence + evidence metadata.

    Modes:
      - "closed_book": answer using the model's own knowledge only.
      - "rag": answer using retrieved evidence only.
      - "rag_confidence": RAG answer + a closed-book pass for agreement-based confidence.

    The final ``answer`` is the RAG answer for rag/rag_confidence, else the closed-book
    answer. This never changes the submission format; it only enriches per-question output.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    evidence_text = ev.format_evidence(evidence_items, char_budget)
    rag = cb = None
    if mode in ("rag", "rag_confidence"):
        rag = answer_one(client, question, options, evidence_text, gen)
    if mode in ("closed_book", "rag_confidence"):
        cb = answer_one(client, question, options, "", gen)

    primary = rag if mode in ("rag", "rag_confidence") else cb
    conf = scoring.compute_confidence(primary, rag, cb, evidence_items, evidence_text, options)

    return {
        "answer": primary["answer"],
        "parsed_from": primary["parsed_from"],
        "raw": primary["raw"],
        "rag_answer": rag["answer"] if rag else None,
        "cb_answer": cb["answer"] if cb else None,
        "confidence": conf["confidence"],
        "components": conf,
        "evidence_quality": scoring.evidence_quality(evidence_items),
        "top_evidence": scoring.top_evidence(evidence_items),
    }
