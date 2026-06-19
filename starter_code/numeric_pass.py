"""Targeted numeric/unit-disambiguation post-pass.

Some multiple-choice options differ only by a number+unit, and a distractor may
reuse a number that belongs to the correct answer in a DIFFERENT unit
(e.g. correct "22.8 miles (36.7 km)" vs distractor "36.7 miles"). A plain LLM
and bag-of-words overlap both get fooled by the shared "36.7" token.

This pass extracts (number, unit) pairs from each option and from the retrieved
evidence, and overrides the current answer ONLY when exactly one option's
number+unit pairs uniquely match the evidence. It is intentionally high
precision: if the signal is ambiguous, the model's answer is left unchanged.

Run AFTER src/run.py (uses the cached evidence + existing submission).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from run import (DEFAULT_OUTPUT_FILE, DEFAULT_QUESTIONS_FILE, load_cache,
                 load_questions, option_texts, rank_passages, to_passages)

ROOT = Path(__file__).resolve().parent.parent
CORRECTIONS_LOG = ROOT / "numeric_corrections.log"

# number (with optional thousands commas / decimals) followed by a unit word,
# separated by space(s) or hyphen(s): "22.8 miles", "22.8-mile", "1,693 ft".
_NUM_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)[\s-]*([A-Za-z]+)")
_UNIT_MAP = {
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
    "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "mile": "mi", "miles": "mi", "ft": "ft", "feet": "ft", "foot": "ft",
}


def norm_unit(u: str) -> str:
    u = u.lower().rstrip(".")
    if u in _UNIT_MAP:
        return _UNIT_MAP[u]
    return u[:-1] if u.endswith("s") and len(u) > 3 else u  # crude singularise


def norm_num(n: str) -> str:
    n = n.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n  # 22.80 -> 22.8


def num_unit_pairs(text: str) -> set:
    return {(norm_num(n), norm_unit(u)) for n, u in _NUM_UNIT.findall(text or "")}


def is_numeric_question(opts: dict) -> bool:
    """Flag questions whose options are dominated by numbers."""
    numeric = sum(1 for v in opts.values() if re.search(r"\d", v))
    return numeric >= max(3, len(opts) - 1)


def numeric_override(row, evidence) -> str | None:
    opts = option_texts(row)
    if not is_numeric_question(opts):
        return None
    passages = rank_passages(row, to_passages(evidence), top_k=10)
    ev_text = " ".join(passages)
    ev_pairs = num_unit_pairs(ev_text)
    if not ev_pairs:
        return None

    scores = {}
    for opt, text in opts.items():
        opt_pairs = num_unit_pairs(text)
        scores[opt] = len(opt_pairs & ev_pairs)

    best = max(scores, key=scores.get)
    top = scores[best]
    # High-precision: need at least one exact pair match and a strict, unique max.
    if top >= 1 and list(scores.values()).count(top) == 1:
        return best
    return None


def main():
    questions = load_questions(DEFAULT_QUESTIONS_FILE)
    cache = load_cache()
    sub = pd.read_csv(DEFAULT_OUTPUT_FILE)
    answers = dict(zip(sub["question_no"].astype(str), sub["answer"]))

    corrections = []
    for _, row in questions.iterrows():
        qno = str(row["question_no"])
        evidence = cache.get(qno)
        if not evidence:
            continue
        suggested = numeric_override(row, evidence)
        if suggested and suggested != answers.get(qno):
            corrections.append((qno, answers.get(qno), suggested,
                                str(row["question"])[:70]))
            answers[qno] = suggested

    sub["answer"] = sub["question_no"].astype(str).map(answers)
    sub.to_csv(DEFAULT_OUTPUT_FILE, index=False)

    with CORRECTIONS_LOG.open("w") as f:
        f.write(f"Numeric post-pass: {len(corrections)} override(s)\n")
        for qno, old, new, q in corrections:
            line = f"Q{qno}: {old} -> {new}   ({q})"
            print(line)
            f.write(line + "\n")
    if not corrections:
        print("Numeric post-pass: no confident overrides.")


if __name__ == "__main__":
    main()
