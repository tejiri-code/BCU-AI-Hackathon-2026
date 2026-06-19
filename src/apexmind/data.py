"""Load + validate questions and write/validate the submission CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from . import OPTION_LETTERS

VALID_ANSWERS = set(OPTION_LETTERS) | {"Unknown"}


def load_questions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"question_no", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"questions file missing columns: {sorted(missing)}")
    return df


def options_of(row: pd.Series) -> Dict[str, str]:
    """Return {letter: text} for present, non-empty options."""
    out: Dict[str, str] = {}
    for letter in OPTION_LETTERS:
        val = row.get(letter, "")
        if pd.notna(val) and str(val).strip():
            out[letter] = str(val).strip()
    return out


def write_submission(records: List[Dict], path: str | Path) -> pd.DataFrame:
    """Write a [question_no, answer] CSV after normalising/validating answers."""
    df = pd.DataFrame(records)
    df = df[["question_no", "answer"]].copy()
    df["answer"] = df["answer"].apply(_normalise_answer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


GOLD_COLUMNS = [
    "question_no", "question", "A", "B", "C", "D", "E", "gold_answer", "source_url", "notes",
]


def ensure_gold_template(questions_path, gold_path, n: int = 20) -> bool:
    """Create a gold-set template from the first `n` questions if it doesn't exist.

    Returns True if a template was created, False if the file already existed.
    """
    gold_path = Path(gold_path)
    if gold_path.exists():
        return False
    q = load_questions(questions_path).head(n).copy()
    for col in ("A", "B", "C", "D", "E"):
        if col not in q.columns:
            q[col] = ""
    q["gold_answer"] = ""
    q["source_url"] = ""
    q["notes"] = ""
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    q[GOLD_COLUMNS].to_csv(gold_path, index=False)
    return True


def load_gold(gold_path) -> pd.DataFrame:
    """Load the gold set; return rows that have a usable gold_answer (A-E)."""
    df = pd.read_csv(gold_path)
    if "gold_answer" not in df.columns:
        raise ValueError("gold file missing 'gold_answer' column")
    df["gold_answer"] = df["gold_answer"].astype(str).str.strip().str.upper()
    labeled = df[df["gold_answer"].isin(OPTION_LETTERS)].copy()
    return labeled


def _normalise_answer(ans) -> str:
    s = str(ans).strip()
    if s.lower() == "unknown":
        return "Unknown"
    s = s.upper()[:1]
    return s if s in set(OPTION_LETTERS) else "Unknown"


def validate_submission(path: str | Path, expected_rows: int | None = None) -> Dict:
    """Check columns, row count, and that every answer is A-E or Unknown."""
    df = pd.read_csv(path)
    problems: List[str] = []
    if list(df.columns) != ["question_no", "answer"]:
        problems.append(f"columns are {list(df.columns)}, expected ['question_no','answer']")
    if expected_rows is not None and len(df) != expected_rows:
        problems.append(f"row count is {len(df)}, expected {expected_rows}")
    bad = df.loc[~df["answer"].astype(str).isin(VALID_ANSWERS), "answer"].tolist()
    if bad:
        problems.append(f"invalid answers present: {bad[:10]}")
    if df["question_no"].duplicated().any():
        problems.append("duplicate question_no values present")
    return {"ok": not problems, "rows": len(df), "problems": problems}
