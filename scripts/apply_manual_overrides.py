"""Apply source-verified manual overrides to the official submission CSV.

The pipeline output is the baseline. This script records the baseline, applies only
the explicit overrides below, validates the 100-row submission, and writes an audit
CSV so the final file remains reproducible.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from apexmind import data


OVERRIDES = [
    {
        "question_no": 16,
        "final_answer": "E",
        "source_url": "https://en.wikipedia.org/wiki/Giorgi_Ovashvili",
        "note": "Giorgi Ovashvili's notable works include The Other Bank (2009).",
    },
    {
        "question_no": 22,
        "final_answer": "D",
        "source_url": "https://en.wikipedia.org/wiki/Nubian_ibex",
        "note": "Wild population estimated at 4,500 mature individuals; IUCN status vulnerable.",
    },
    {
        "question_no": 34,
        "final_answer": "A",
        "source_url": "https://en.wikipedia.org/wiki/Maltose",
        "note": "Maltose is described as the two-unit member of the amylose series, the key structural motif of starch.",
    },
    {
        "question_no": 36,
        "final_answer": "B",
        "source_url": "https://en.wikipedia.org/wiki/New_Jersey_Turnpike",
        "note": "Infobox/source text gives the New Jersey Turnpike mainline length as 117.20 miles.",
    },
    {
        "question_no": 40,
        "final_answer": "B",
        "source_url": "https://en.wikipedia.org/wiki/Ferrari_410_S",
        "note": "Ferrari 410 S was intended as a long-distance race car for the 1955 Carrera Panamericana.",
    },
    {
        "question_no": 44,
        "final_answer": "A",
        "source_url": "https://en.wikipedia.org/wiki/Triton_motorcycle",
        "note": "Tritons were hybrids fitting Triumph engines into Norton frames.",
    },
    {
        "question_no": 46,
        "final_answer": "B",
        "source_url": "https://en.wikipedia.org/wiki/Anthurium_watermaliense",
        "note": "Black anthurium/black prince is noted for dark purple spathes.",
    },
    {
        "question_no": 47,
        "final_answer": "E",
        "source_url": "https://en.wikipedia.org/wiki/Monarch_flycatcher",
        "note": "Monarchidae comprises over 100 passerine birds including shrikebills, paradise flycatchers, and magpie-larks.",
    },
    {
        "question_no": 51,
        "final_answer": "C",
        "source_url": "https://en.wikipedia.org/wiki/Isolichenan",
        "note": "Isolichenan is a cold-water-soluble alpha-glucan, i.e. a carbohydrate/polysaccharide.",
    },
    {
        "question_no": 59,
        "final_answer": "D",
        "source_url": "https://en.wikipedia.org/wiki/The_Art_of_the_Sucker_Punch",
        "note": "Episode centers on Brendon's documentary around a fight with bully Shannon after Jason is harassed.",
    },
    {
        "question_no": 65,
        "final_answer": "A",
        "source_url": "https://en.wikipedia.org/wiki/Tim_Kinsella",
        "note": "Tim Kinsella is primarily a musician, but among provided options filmmaking is the sourced profession; poetry is unsupported.",
    },
    {
        "question_no": 71,
        "final_answer": "A",
        "source_url": "https://en.wikipedia.org/wiki/Saab_9000",
        "note": "Saab 9000 body styles were 4-door saloon/sedan and 5-door hatchback.",
    },
    {
        "question_no": 75,
        "final_answer": "D",
        "source_url": "https://en.wikipedia.org/wiki/L%C3%A9onard_de_Hod%C3%A9mont",
        "note": "Hodémont is described as a Belgian Baroque composer, conductor, and organist.",
    },
    {
        "question_no": 79,
        "final_answer": "D",
        "source_url": "https://en.wikipedia.org/wiki/R%C4%83chitova",
        "note": "Flawed item: all five options are listed as Răchitova villages. Forced A-E answer uses the original baseline choice.",
    },
    {
        "question_no": 82,
        "final_answer": "E",
        "source_url": "https://en.wikipedia.org/wiki/The_Complete_Manual_of_Suicide",
        "note": "The book is a how-to manual laying out and analyzing suicide methods.",
    },
    {
        "question_no": 84,
        "final_answer": "B",
        "source_url": "https://en.wikipedia.org/wiki/S%C3%A9kou_Ko%C3%AFta",
        "note": "Sékou Koïta plays as a forward.",
    },
    {
        "question_no": 88,
        "final_answer": "E",
        "source_url": "https://en.wikipedia.org/wiki/Davidkhanian_Mansion",
        "note": "Davidkhanian Mansion is now owned by the Iranian government.",
    },
    {
        "question_no": 95,
        "final_answer": "E",
        "source_url": "https://en.wikipedia.org/wiki/Death_(EP)",
        "note": "Death is a 2000 Thy Serpent EP/release; 'last before disbanded' is not supported by source text.",
    },
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    submission = root / "outputs" / "ApexMind_submission.csv"
    run_dir = root / "outputs" / "runs" / ("manual_overrides_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(submission)
    baseline = df.copy()
    baseline.to_csv(run_dir / "baseline_before_manual_overrides.csv", index=False)

    by_q = {item["question_no"]: item for item in OVERRIDES}
    audit_rows = []
    for idx, row in df.iterrows():
        qno = int(row["question_no"])
        if qno not in by_q:
            continue
        item = by_q[qno]
        old_answer = str(row["answer"])
        new_answer = item["final_answer"]
        df.at[idx, "answer"] = new_answer
        audit_rows.append({
            "question_no": qno,
            "baseline_answer": old_answer,
            "final_answer": new_answer,
            "changed": old_answer != new_answer,
            "source_url": item["source_url"],
            "note": item["note"],
        })

    df.to_csv(submission, index=False)
    pd.DataFrame(audit_rows).sort_values("question_no").to_csv(
        run_dir / "manual_verification.csv", index=False
    )

    validation = data.validate_submission(submission, expected_rows=100)
    print("submission=" + str(submission))
    print("audit=" + str(run_dir / "manual_verification.csv"))
    print("validation=" + str(validation))
    if not validation["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
