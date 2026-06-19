"""Build the FINAL ApexMind submission from the reconciled, human-verified answer key.

Reconciliation policy (transparent + reproducible):
  1. Spine = the team's deep-research answer key (manually compiled from Wikipedia +
     the "Comprehensive Verification Report" PDF).
  2. Overrides = the team's *sourced* manual verification (final_manual_verification.csv,
     every row carries a Wikipedia source_url). Applied where it contradicts the spine,
     because those rows are individually source-audited.

Only 5 questions differ between the two sources; all overrides are listed below with
their source so any single answer can be flipped with a one-line edit.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ApexMind_submission.csv"

# 1) Deep-research key (1-100), compiled by the team from primary sources.
DEEP_RESEARCH = {
1:"B",2:"D",3:"B",4:"D",5:"D",6:"A",7:"E",8:"A",9:"C",10:"E",11:"A",12:"B",13:"C",14:"B",
15:"E",16:"E",17:"C",18:"E",19:"E",20:"B",21:"B",22:"D",23:"C",24:"B",25:"A",26:"A",27:"C",
28:"A",29:"C",30:"B",31:"A",32:"B",33:"B",34:"A",35:"D",36:"E",37:"A",38:"C",39:"A",40:"B",
41:"C",42:"A",43:"C",44:"A",45:"B",46:"B",47:"E",48:"D",49:"C",50:"B",51:"C",52:"A",53:"A",
54:"E",55:"E",56:"D",57:"A",58:"B",59:"D",60:"D",61:"E",62:"E",63:"C",64:"D",65:"D",66:"C",
67:"B",68:"A",69:"C",70:"B",71:"A",72:"B",73:"A",74:"E",75:"D",76:"B",77:"D",78:"C",79:"A",
80:"A",81:"E",82:"E",83:"A",84:"B",85:"A",86:"D",87:"E",88:"C",89:"B",90:"A",91:"A",92:"B",
93:"B",94:"D",95:"A",96:"E",97:"A",98:"C",99:"C",100:"B"}

# 2) Sourced manual-verification overrides (Wikipedia URL per row). These win on conflict.
OVERRIDES = {
    16: ("E", "Giorgi Ovashvili -> The Other Bank (2009)", "en.wikipedia.org/wiki/Giorgi_Ovashvili"),
    36: ("B", "New Jersey Turnpike mainline 117.20 mi (closest valid option)", "en.wikipedia.org/wiki/New_Jersey_Turnpike"),
    65: ("A", "Tim Kinsella - filmmaking is the sourced profession", "en.wikipedia.org/wiki/Tim_Kinsella"),
    79: ("Unknown", "Rachitova - flawed item: all five options are Rachitova villages; no single correct option", "en.wikipedia.org/wiki/R%C4%83chitova"),
    88: ("E", "Davidkhanian Mansion now owned by the Iranian government", "en.wikipedia.org/wiki/Davidkhanian_Mansion"),
    95: ("E", "Death (2000 Thy Serpent EP)", "en.wikipedia.org/wiki/Death_(EP)"),
}

VALID = {"A", "B", "C", "D", "E", "Unknown"}  # rules allow 'Unknown' for flawed items


def main() -> None:
    final = dict(DEEP_RESEARCH)
    applied = []
    for q, (ans, why, url) in OVERRIDES.items():
        if final.get(q) != ans:
            applied.append((q, final.get(q), ans, why, url))
        final[q] = ans

    assert sorted(final) == list(range(1, 101)), "key must cover 1..100"
    assert all(v in VALID for v in final.values()), "all answers must be A-E"

    df = pd.DataFrame({"question_no": list(range(1, 101)),
                       "answer": [final[q] for q in range(1, 101)]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Wrote {OUT} ({len(df)} rows).")
    print(f"\nOverrides applied vs deep-research key ({len(applied)}):")
    for q, old, new, why, url in applied:
        print(f"  Q{q}: {old} -> {new}   [{why}]  ({url})")
    dist = df["answer"].value_counts().reindex(list("ABCDE")).fillna(0).astype(int)
    print("\nAnswer distribution:", dict(dist))


if __name__ == "__main__":
    main()
