"""
Sanity checks for VeriQuest AI (no network/LLM calls - pure logic tests).

Run with:
    python3 -m unittest tests/test_pipeline.py -v
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter_code"))

import run  # noqa: E402


class TestRepairMojibake(unittest.TestCase):
    def test_fixes_latin1_double_encoding(self):
        self.assertEqual(run.repair_mojibake("cafÃ©"), "café")
        self.assertEqual(run.repair_mojibake("Le prophÃ¨te"), "Le prophète")

    def test_fixes_cp1252_double_encoding(self):
        # byte 0x83 mis-decoded via Windows-1252 as the florin sign "ƒ"
        self.assertEqual(run.repair_mojibake("RÄƒchitova"), "Răchitova")

    def test_leaves_correct_text_unchanged(self):
        self.assertEqual(run.repair_mojibake("plain ascii text"), "plain ascii text")
        self.assertEqual(run.repair_mojibake("Klokočná"), "Klokočná")

    def test_handles_none(self):
        self.assertIsNone(run.repair_mojibake(None))


class TestCleanAnswer(unittest.TestCase):
    def test_accepts_bare_letter(self):
        self.assertEqual(run.clean_answer("B"), "B")
        self.assertEqual(run.clean_answer("b"), "B")

    def test_handles_sentence_response(self):
        self.assertEqual(run.clean_answer("The answer is B."), "B")
        self.assertEqual(run.clean_answer("Answer: D"), "D")

    def test_handles_unknown(self):
        self.assertEqual(run.clean_answer("Unknown"), "Unknown")
        self.assertEqual(run.clean_answer("I don't know, unknown"), "Unknown")

    def test_never_returns_blank(self):
        self.assertEqual(run.clean_answer(""), "Unknown")
        self.assertEqual(run.clean_answer(None), "Unknown")
        self.assertEqual(run.clean_answer("   "), "Unknown")

    def test_rejects_invalid_letters(self):
        self.assertEqual(run.clean_answer("Z"), "Unknown")
        self.assertEqual(run.clean_answer("F. some text"), "Unknown")

    def test_only_allowed_values(self):
        for raw in ["A", "b", "The answer is C", "  d  ", "garbage", "unknown"]:
            self.assertIn(run.clean_answer(raw), run.ALLOWED_ANSWERS)


class TestBuildSearchQueries(unittest.TestCase):
    SAMPLE_ROW = {
        "question_no": "1",
        "question": "What was P. Padmarajan's contribution to Malayalam cinema?",
        "A": "P. Padmarajan was a noted film critic.",
        "B": "P. Padmarajan was an Indian film maker, screenwriter and author.",
        "C": "P. Padmarajan was a pioneer in special effects.",
        "D": "P. Padmarajan was a renowned cinematographer.",
        "E": "P. Padmarajan popularized non-linear storytelling.",
    }

    def test_returns_multiple_queries(self):
        queries = run.build_search_queries(self.SAMPLE_ROW)
        self.assertGreaterEqual(len(queries), 3)
        self.assertLessEqual(len(queries), 6)

    def test_does_not_dump_all_options_into_one_query(self):
        queries = run.build_search_queries(self.SAMPLE_ROW)
        all_option_text = " ".join(self.SAMPLE_ROW[letter] for letter in run.OPTION_LETTERS)
        for q in queries:
            # no single query should contain every option's text concatenated
            self.assertNotEqual(q.strip(), (self.SAMPLE_ROW["question"] + " " + all_option_text).strip())
            self.assertLess(len(q), len(all_option_text))

    def test_queries_are_deduplicated(self):
        queries = run.build_search_queries(self.SAMPLE_ROW)
        lowered = [q.lower() for q in queries]
        self.assertEqual(len(lowered), len(set(lowered)))

    def test_handles_missing_options_gracefully(self):
        row = {"question_no": "2", "question": "Who wrote Hamlet?", "A": "William Shakespeare", "B": "", "C": "", "D": "", "E": ""}
        queries = run.build_search_queries(row)
        self.assertGreaterEqual(len(queries), 1)


class TestDeterministicFallback(unittest.TestCase):
    def test_returns_valid_answer_with_evidence(self):
        options = {
            "A": "The Agadez Mosque is made of concrete.",
            "B": "The Agadez Mosque is made of glass.",
            "C": "The Agadez Mosque is made of bamboo.",
            "D": "The Agadez Mosque is made of stone.",
            "E": "The Agadez Mosque is made of clay.",
        }
        evidence = [
            {
                "title": "Agadez Mosque - Wikipedia",
                "snippet": "The Great Mosque of Agadez is built of clay and adobe in the Sudano-Sahelian style.",
                "url": "https://en.wikipedia.org/wiki/Agadez_Mosque",
            }
        ]
        ranked = run.rank_evidence({"question": "What is the Agadez Mosque made of?"}, evidence, options)
        option_scores, exact_flags = run.score_options(options, ranked)
        answer = run.best_deterministic_answer(option_scores)
        self.assertIn(answer, run.ALLOWED_ANSWERS)
        self.assertEqual(answer, "E")

    def test_returns_unknown_with_no_evidence(self):
        options = {"A": "Option A text", "B": "Option B text"}
        option_scores, _ = run.score_options(options, [])
        answer = run.best_deterministic_answer(option_scores)
        self.assertEqual(answer, "Unknown")

    def test_ignores_numbers_shared_with_the_question(self):
        # Regression test: "U.S. Route 11" repeats the number "11" in nearly
        # every search result, which used to spuriously inflate option A
        # ("11 miles...") even though the real answer (D) was stated
        # explicitly in the top evidence ("22.8-mile-long (36.7 km)").
        question = "What is the length of U.S. Route 11 in Georgia?"
        options = {
            "A": "11 miles (17.7 km)",
            "B": "36.7 miles (58.9 km)",
            "C": "58 miles (93.3 km)",
            "D": "22.8 miles (36.7 km)",
            "E": "8 miles (12.9 km)",
        }
        evidence = [
            {
                "title": "U.S. Route 11 in Georgia - Wikipedia",
                "snippet": "U.S. Highway 11 (US 11) in the U.S. state of Georgia is a "
                "22.8-mile-long (36.7 km) United States Numbered Highway "
                "that travels north-south through Dade County.",
                "url": "https://en.wikipedia.org/wiki/U.S._Route_11_in_Georgia",
            }
        ]
        ranked = run.rank_evidence({"question": question}, evidence, options)
        option_scores, exact_flags = run.score_options(options, ranked, question)
        self.assertEqual(run.best_deterministic_answer(option_scores), "D")
        self.assertTrue(exact_flags["D"])

    def test_confidence_penalises_ties(self):
        tied_scores = {"A": 1.0, "B": 1.0, "C": 0.2, "D": 0.0, "E": 0.0}
        clear_scores = {"A": 1.0, "B": 0.1, "C": 0.05, "D": 0.0, "E": 0.0}
        tied_confidence = run.compute_confidence(tied_scores, [{}, {}, {}], {"A": False, "B": False})
        clear_confidence = run.compute_confidence(clear_scores, [{}, {}, {}], {"A": True})
        self.assertLess(tied_confidence, clear_confidence)
        self.assertLess(tied_confidence, run.HIGH_CONFIDENCE_THRESHOLD)


class TestCsvOutputFormat(unittest.TestCase):
    def test_write_submission_csv_has_exact_columns(self):
        predictions = [{"question_no": "1", "answer": "A"}, {"question_no": "2", "answer": "Unknown"}]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.csv")
            run.write_submission_csv(predictions, out_path)
            with open(out_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ["question_no", "answer"])
                rows = list(reader)
                self.assertEqual(rows, [["1", "A"], ["2", "Unknown"]])

    def test_validate_submission_rejects_invalid_answers(self):
        with self.assertRaises(ValueError):
            run.validate_submission([{"question_no": "1", "answer": "Z"}], limit=None)

    def test_validate_submission_rejects_blank_answers(self):
        with self.assertRaises(ValueError):
            run.validate_submission([{"question_no": "1", "answer": ""}], limit=None)

    def test_validate_submission_accepts_valid_rows(self):
        predictions = [{"question_no": str(i), "answer": "A"} for i in range(1, 101)]
        run.validate_submission(predictions, limit=None)  # should not raise


if __name__ == "__main__":
    unittest.main()
