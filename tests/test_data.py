import pandas as pd

from apexmind import data as d


def test_options_of_skips_blank():
    row = pd.Series({"A": "x", "B": " ", "C": "y", "D": None})
    assert d.options_of(row) == {"A": "x", "C": "y"}


def test_write_and_validate_submission(tmp_path):
    recs = [{"question_no": i, "answer": a} for i, a in enumerate(["A", "b", "Unknown", "Z", "C"], 1)]
    out = tmp_path / "sub.csv"
    df = d.write_submission(recs, out)
    # 'b' -> 'B', 'Z' -> 'Unknown'
    assert df["answer"].tolist() == ["A", "B", "Unknown", "Unknown", "C"]
    val = d.validate_submission(out, expected_rows=5)
    assert val["ok"], val["problems"]
