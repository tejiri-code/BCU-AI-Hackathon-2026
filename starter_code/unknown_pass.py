"""Unknown-detection post-pass.

`Unknown` is an allowed answer and is the *correct* one when none of the listed
options can be right — most reliably on NEGATION questions ("which is NOT...",
"...EXCEPT", "least...") where the evidence shows that EVERY option actually
holds, so there is no valid exception (e.g. Q79: all five villages really are
part of the commune).

Important: with no negative marking, a guess (20% chance) beats `Unknown` (0%),
so `Unknown` must only be used when it is genuinely correct — never as a
low-confidence cop-out. We therefore scope auto-conversion to negation questions
and require an evidence-grounded LLM to confirm that none of the options is the
exception. Non-negation questions are left untouched.

Run AFTER run.py (uses cached evidence + the existing submission).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from run import (DEFAULT_OUTPUT_FILE, DEFAULT_QUESTIONS_FILE, ask_llm,
                 load_cache, load_questions, option_texts, parse_letter,
                 rank_passages, to_passages)

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "unknown_pass.log"

# A genuine negation / exception question.
NEGATION_RE = re.compile(r"\bNOT\b|\bEXCEPT\b|\bleast\b|\bnever\b|\bno longer\b|\bincorrect\b|\bfalse\b")

PROMPT = """You are verifying a multiple-choice question against EVIDENCE.

This question asks which option is the EXCEPTION (e.g. which is NOT true / does
NOT belong). The answer is the single option that is FALSE. BUT if the evidence
shows that EVERY option is actually true — so none of them is the exception —
then respond "Unknown".

EVIDENCE:
{ev}

QUESTION:
{q}

OPTIONS:
{opts}

Respond with exactly one token: a single letter A-E, or the word Unknown."""


def is_unknown(raw: str | None) -> bool:
    return bool(raw) and "unknown" in raw.strip().lower()


def main():
    questions = load_questions(DEFAULT_QUESTIONS_FILE)
    cache = load_cache()
    sub = pd.read_csv(DEFAULT_OUTPUT_FILE)
    answers = dict(zip(sub["question_no"].astype(int), sub["answer"]))

    changes = []
    for _, row in questions.iterrows():
        qno = int(row["question_no"])
        if not NEGATION_RE.search(str(row["question"])):
            continue  # only negation questions are eligible for auto-Unknown
        opts = option_texts(row)
        passages = rank_passages(row, to_passages(cache.get(str(qno), [])))
        ev = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages)) or "(none)"
        raw = ask_llm(PROMPT.format(ev=ev, q=row["question"],
                                    opts="\n".join(f"{k}. {v}" for k, v in opts.items())))
        if is_unknown(raw) and answers.get(qno) != "Unknown":
            changes.append((qno, answers.get(qno)))
            answers[qno] = "Unknown"

    sub["answer"] = sub["question_no"].astype(int).map(answers)
    sub.to_csv(DEFAULT_OUTPUT_FILE, index=False)

    with LOG.open("w") as f:
        f.write(f"Unknown pass: {len(changes)} question(s) set to Unknown\n")
        for qno, old in changes:
            line = f"Q{qno}: {old} -> Unknown"
            print(line)
            f.write(line + "\n")
    if not changes:
        print("Unknown pass: no questions set to Unknown.")


if __name__ == "__main__":
    main()
