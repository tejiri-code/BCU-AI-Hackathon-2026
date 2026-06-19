"""MCQ prompt construction (RAG and closed-book variants)."""
from __future__ import annotations

from typing import Dict, List

SYSTEM = (
    "You are a meticulous question-answering expert. You answer multiple-choice "
    "questions by selecting the single best option. Base your answer on the provided "
    "evidence and your knowledge. Think briefly, then end with a line of the exact form "
    "'Answer: X' where X is one of A, B, C, D, or E."
)


def _format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def build_messages(
    question: str,
    options: Dict[str, str],
    evidence: str = "",
    disable_thinking: bool = True,
) -> List[Dict[str, str]]:
    """Build chat messages. If evidence is empty this is a closed-book prompt."""
    parts = [f"Question:\n{question}\n", f"Options:\n{_format_options(options)}\n"]
    if evidence.strip():
        parts.append(f"Evidence (may be partial or noisy; use what is relevant):\n{evidence}\n")
    parts.append(
        "Choose the single best option. Respond with a short justification (one or two "
        "sentences) followed by a final line 'Answer: X'."
    )
    user = "\n".join(parts)

    # Qwen3 is a hybrid reasoning model; '/no_think' keeps the primary pass fast + deterministic.
    system = SYSTEM + (" /no_think" if disable_thinking else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
