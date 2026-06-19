"""
BCU AI Hackathon 2026 - Participant Starter Code
Laptop / local Python version

Goal:
    Build a Generative AI question-answering pipeline for 100 multiple-choice questions.

What this starter code provides:
    1. Load the official questions CSV.
    2. Retrieve web evidence using DuckDuckGo.
    3. Provide TODO placeholders for evidence ranking and answer generation.
    4. Export the final answer CSV in the required format.

What participants should improve:
    - Better query construction
    - Evidence ranking
    - RAG prompt construction
    - LLM answer generation
    - Answer validation and error handling

Model rule:
    Any selected LLM must not exceed 8B parameters.
"""

# First-time setup:
# Windows:
# py -m pip install -r requirements.txt

# Mac/Linux:
# python3 -m pip install -r requirements.txt

import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = BASE_DIR.parent / "questions_100.csv"
DEFAULT_OUTPUT_FILE = "TEAMNAME_submission.csv"

ENABLE_WEB_SEARCH = True
MAX_RESULTS_PER_QUERY = 5
SEARCH_SLEEP_SECONDS = 0.5

ALLOWED_ANSWERS = {"A", "B", "C", "D", "E", "Unknown"}


# -----------------------------
# Data loading
# -----------------------------

def load_questions(path: str) -> pd.DataFrame:
    """Load the official question file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place the file in the same folder as this script "
            "or pass --questions with the correct path."
        )

    questions = pd.read_csv(file_path)

    required_columns = {"question_no", "question"}
    missing = required_columns - set(questions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return questions


# -----------------------------
# Query construction
# -----------------------------

def build_search_query(row: pd.Series) -> str:
    """
    Build a search query from the question and options.

    TODO: Improve this function.
    Ideas:
        - Add important keywords from the options.
        - Remove unnecessary words.
        - Generate multiple queries per question.
        - Use domain-specific keywords.
    """
    question = str(row.get("question", "")).strip()

    option_texts = []
    for option in ["A", "B", "C", "D", "E"]:
        value = row.get(option, "")
        if pd.notna(value) and str(value).strip():
            option_texts.append(str(value).strip())

    return " ".join([question] + option_texts)


# -----------------------------
# Evidence retrieval
# -----------------------------

def search_duckduckgo(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict[str, str]]:
    """Retrieve evidence snippets from DuckDuckGo."""
    if not DDGS_AVAILABLE:
        print("[Warning] ddgs not installed. Returning empty evidence.")
        return []

    if not ENABLE_WEB_SEARCH:
        return []

    evidence = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for item in results:
                evidence.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("body", ""),
                        "url": item.get("href", ""),
                    }
                )
    except Exception as error:
        print(f"[Warning] DuckDuckGo search failed: {error}")

    time.sleep(SEARCH_SLEEP_SECONDS)
    return evidence


# -----------------------------
# Evidence ranking - TODO
# -----------------------------

def rank_evidence(row: pd.Series, evidence: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, str]]:
    """
    Rank retrieved evidence and return the best snippets.

    TODO: Implement your ranking method.
    Possible approaches:
        - Keyword overlap
        - TF-IDF
        - BM25
        - Sentence embeddings
        - Cross-encoder reranking
        - LLM-based evidence selection

    Starter behaviour:
        This placeholder simply returns the first top_k evidence items.
    """
    return evidence[:top_k]


# -----------------------------
# Prompt building - TODO
# -----------------------------

def build_prompt(row: pd.Series, ranked_evidence: List[Dict[str, str]]) -> str:
    """
    Build a prompt for your chosen LLM.

    TODO: Improve this prompt.
    The final model should return only one letter: A, B, C, D, or E.
    """
    question = str(row.get("question", ""))

    options = []
    for option in ["A", "B", "C", "D", "E"]:
        value = row.get(option, "")
        if pd.notna(value) and str(value).strip():
            options.append(f"{option}. {value}")

    evidence_text = "\n".join(
        [f"- {item.get('title', '')}: {item.get('snippet', '')}" for item in ranked_evidence]
    )

    prompt = f"""
Answer the following multiple-choice question.
Use the evidence if relevant.
Return only one letter: A, B, C, D, or E.

Question:
{question}

Options:
{chr(10).join(options)}

Evidence:
{evidence_text}

Answer:
""".strip()
    return prompt


# -----------------------------
# Answer generation - TODO
# -----------------------------

def generate_answer(prompt: str) -> str:
    """
    Generate the final answer using your chosen LLM.

    TODO: Replace this placeholder with your own model/API/local LLM.
    Examples:
        - Local Hugging Face model <= 8B parameters
        - OpenAI/Gemini/Claude API, if allowed by organiser rules
        - A small instruction-tuned model
        - A retrieval-only heuristic baseline

    Starter behaviour:
        This placeholder returns 'A' so the pipeline can run end-to-end.
    """
    return "A"


def clean_answer(answer: str) -> str:
    """Validate and normalise the answer."""
    answer = str(answer).strip()

    if answer.lower() == "unknown":
        return "Unknown"

    answer = answer.upper()[:1]
    if answer in {"A", "B", "C", "D", "E"}:
        return answer

    return "Unknown"


# -----------------------------
# Main pipeline
# -----------------------------

def answer_question(row: pd.Series) -> Dict[str, str]:
    """Run the full starter pipeline for one question."""
    query = build_search_query(row)
    evidence = search_duckduckgo(query)
    ranked_evidence = rank_evidence(row, evidence)
    prompt = build_prompt(row, ranked_evidence)
    answer = clean_answer(generate_answer(prompt))

    return {
        "question_no": row.get("question_no"),
        "answer": answer,
    }


def run_pipeline(questions_file: str, output_file: str, limit: int | None = None) -> pd.DataFrame:
    questions = load_questions(questions_file)

    if limit is not None:
        questions = questions.head(limit)

    predictions = []
    for index, row in questions.iterrows():
        question_no = row.get("question_no", index + 1)
        print(f"Answering question {question_no}...")
        predictions.append(answer_question(row))

    submission = pd.DataFrame(predictions, columns=["question_no", "answer"])
    submission.to_csv(output_file, index=False)
    print(f"\nSaved submission file to: {output_file}")
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCU AI Hackathon 2026 starter code")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_FILE, help="Path to questions_100.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output CSV filename")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of questions to test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.questions, args.output, args.limit)

"""
Suggested improvements

Try improving the TODO sections step by step.

Suggested order:
1. Test DuckDuckGo retrieval on 3–5 questions
2. Improve search query generation
3. Improve evidence ranking
4. Add an LLM answer generation method
5. Add confidence scoring
6. Add logging so you can inspect evidence and answers
7. Test on questions
8. Run on the full question set

Remember: the final submitted answers must be `A`, `B`, `C`, `D`, or `E`.
"""