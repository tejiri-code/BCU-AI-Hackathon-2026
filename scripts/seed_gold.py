"""Seed data/gold_dev.csv from the first 20 questions, pre-filling a verified subset.

Only questions verified against Wikipedia with high confidence are labeled; the rest
are left blank for the team to fill. Re-running will NOT overwrite an existing file.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "questions_100.csv"
GOLD = ROOT / "data" / "gold_dev.csv"
COLUMNS = ["question_no", "question", "A", "B", "C", "D", "E", "gold_answer", "source_url", "notes"]

# (gold_answer, source_url, notes) for verified questions. Blank rows stay for the team.
VERIFIED = {
    1:  ("B", "https://en.wikipedia.org/wiki/Marine_Raider_Regiment", "MARSOC: direct action, special recon, FID"),
    2:  ("D", "https://en.wikipedia.org/wiki/George_Anson_(British_Army_officer)", "Served under Wellington in the Peninsular War (Napoleonic Wars)"),
    3:  ("B", "https://en.wikipedia.org/wiki/P._Padmarajan", "Filmmaker/screenwriter/author; new school of Malayalam cinema, 1980s"),
    5:  ("D", "https://en.wikipedia.org/wiki/Edmond_Malone", "Shakespearean scholar and editor"),
    9:  ("C", "https://en.wikipedia.org/wiki/CoRoT", "Searched for exoplanets with short orbital periods"),
    12: ("B", "https://en.wikipedia.org/wiki/Volga-Dnepr_Airlines", "Petrochemical, energy, aerospace, agriculture, telecom (moderate confidence)"),
    13: ("C", "https://en.wikipedia.org/wiki/Scum_of_the_Earth_Church", "Named from 1 Corinthians 4:11-13"),
    14: ("B", "https://en.wikipedia.org/wiki/General_Electric_F414", "Powers military fighter jets (F/A-18E/F, Gripen E)"),
    16: ("E", "https://en.wikipedia.org/wiki/Giorgi_Ovashvili", "The Other Bank (2009)"),
    17: ("C", "https://en.wikipedia.org/wiki/Are_You_in_Love%3F_(Basia_Bulat_album)", "Released on Secret City Records"),
}


def main() -> None:
    if GOLD.exists():
        print(f"{GOLD} already exists; not overwriting.")
        return
    q = pd.read_csv(QUESTIONS).head(20).copy()
    for col in ("A", "B", "C", "D", "E"):
        if col not in q.columns:
            q[col] = ""
    q["gold_answer"] = ""
    q["source_url"] = ""
    q["notes"] = ""
    for qno, (ans, url, note) in VERIFIED.items():
        mask = q["question_no"] == qno
        q.loc[mask, "gold_answer"] = ans
        q.loc[mask, "source_url"] = url
        q.loc[mask, "notes"] = note
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    q[COLUMNS].to_csv(GOLD, index=False)
    print(f"Wrote {GOLD} with {len(VERIFIED)} verified / {len(q)} total rows.")


if __name__ == "__main__":
    main()
