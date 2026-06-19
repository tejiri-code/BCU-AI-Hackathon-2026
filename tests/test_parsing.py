from apexmind.parsing import parse_letter, match_by_option_text


def test_answer_colon():
    assert parse_letter("Some reasoning.\nAnswer: C") == "C"


def test_answer_is_parenthetical():
    assert parse_letter("I believe the answer is (B).") == "B"


def test_lone_letter_line():
    assert parse_letter("D") == "D"


def test_prefers_final_restatement():
    assert parse_letter("Maybe A at first, but Answer: E") == "E"


def test_none_when_absent():
    assert parse_letter("no valid choice here 123") is None


def test_option_text_fallback():
    options = {"A": "Paris", "B": "Berlin", "C": "Madrid"}
    assert match_by_option_text("The capital is Berlin, clearly.", options) == "B"
