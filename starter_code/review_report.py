"""
VeriQuest AI - manual review helper.

There is no legitimate ground-truth checker available to us (the hackathon
rules forbid using/possessing the organiser answer key), so the only honest
way to "confirm" answers is to manually spot-check the riskiest ones against
the real evidence. This script reads logs/evidence_log.jsonl and prints the
questions most likely to be wrong, ranked by:
    1. low confidence
    2. disagreement between the LLM judge and the deterministic scorer
so you can quickly click through the listed evidence URLs yourself.

Usage:
    python starter_code/review_report.py [--top N] [--threshold 0.6]
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = BASE_DIR.parent / "logs" / "evidence_log.jsonl"


def load_log(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def risk_score(row) -> float:
    """Lower = riskier. Disagreement between LLM and deterministic scorer is
    the strongest red flag; low confidence is the second."""
    disagreement_penalty = 0.0
    if row["llm_answer"] != row["deterministic_answer"]:
        disagreement_penalty = 0.5
    return row["confidence"] - disagreement_penalty


def main():
    parser = argparse.ArgumentParser(description="Flag the riskiest VeriQuest AI answers for manual review")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Path to evidence_log.jsonl")
    parser.add_argument("--top", type=int, default=20, help="How many risky questions to show")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"No log found at {log_path}. Run starter_code/run.py first.")
        return

    rows = load_log(log_path)
    rows.sort(key=risk_score)

    print(f"Loaded {len(rows)} answered questions from {log_path}")
    print(f"Showing the {min(args.top, len(rows))} riskiest answers (lowest confidence / LLM-vs-deterministic disagreement).")
    print("Open the listed URL(s) yourself and check whether the evidence actually supports the final answer.\n")

    for row in rows[: args.top]:
        agree = "AGREE" if row["llm_answer"] == row["deterministic_answer"] else "DISAGREE"
        print("=" * 90)
        print(f"Q{row['question_no']}  final={row['final_answer']}  confidence={row['confidence']}  "
              f"llm={row['llm_answer']}  deterministic={row['deterministic_answer']}  [{agree}]")
        for i, ev in enumerate(row["ranked_evidence"][:3], start=1):
            print(f"  [{i}] {ev.get('title', '')}")
            print(f"      {ev.get('url', '')}")
        print()


if __name__ == "__main__":
    main()
