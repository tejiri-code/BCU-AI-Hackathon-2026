"""
VeriQuest AI - BCU AI Hackathon 2026
Evidence-grounded multiple-choice question answering pipeline.

Pipeline:
    question -> multiple targeted search queries -> DuckDuckGo evidence retrieval
    -> evidence deduplication -> evidence ranking -> deterministic option scoring
    -> RAG prompt -> <=8B local LLM judge (Ollama) -> deterministic fallback
    -> strict answer validation -> final CSV export -> debug logs

Model rule: the LLM used for answering must be <=8B parameters. The default
model is llama3.1:8b (8.0B, Q4_K_M) served locally via Ollama. Override with:
    export HACKATHON_MODEL=qwen2.5:7b-instruct
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = BASE_DIR.parent / "questions_100.csv"
DEFAULT_OUTPUT_FILE = "TEAMNAME_submission.csv"
LOGS_DIR = BASE_DIR.parent / "logs"

ENABLE_WEB_SEARCH = True
MAX_RESULTS_PER_QUERY = 5
MAX_QUERIES_PER_QUESTION = 6
SEARCH_SLEEP_SECONDS = 0.4
TOP_K_EVIDENCE = 5

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "llama3.1:8b"  # 8.0B parameters, Q4_K_M - satisfies the <=8B rule
OLLAMA_TIMEOUT_SECONDS = 30
DISABLE_LLM = False  # set by --no-llm

ALLOWED_ANSWERS = {"A", "B", "C", "D", "E", "Unknown"}
OPTION_LETTERS = ["A", "B", "C", "D", "E"]

HIGH_CONFIDENCE_THRESHOLD = 0.55

AUTHORITATIVE_DOMAINS = {
    "wikipedia.org": 1.0,
    "britannica.com": 0.6,
    "imdb.com": 0.5,
    "biography.com": 0.4,
    "bbc.co.uk": 0.3,
    "bbc.com": 0.3,
}

GENERAL_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "by", "with", "and",
    "or", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "it", "its", "as", "from", "which", "who", "whom",
    "what", "when", "where", "why", "how", "do", "does", "did", "has", "have",
    "had", "will", "would", "can", "could", "should", "may", "might", "must",
    "not", "no", "yes", "than", "then", "also", "into", "about", "over",
    "under", "between", "among", "during", "after", "before", "up", "down",
    "out", "off", "again", "further", "each", "other", "some", "such", "only",
    "own", "same", "so", "too", "very", "just", "now", "according",
}

QUESTION_LEADING_STOPWORDS = {
    "what", "who", "when", "where", "which", "why", "how", "is", "are", "was",
    "were", "do", "does", "did", "the", "a", "an", "of", "in", "on", "at",
    "for", "to", "by", "with", "according", "wikipedia", "and", "or",
}

_ddgs_warned = False
_ollama_warned = False


# -----------------------------
# Data loading
# -----------------------------

def load_questions(path: str) -> List[Dict[str, str]]:
    """Load the official question file using the stdlib csv module.

    pandas/numpy fail to build on pre-release Python (e.g. 3.15 alpha) since
    no wheels exist yet, so this pipeline avoids that dependency entirely.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place the file in the same folder as this script "
            "or pass --questions with the correct path."
        )

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [{k: repair_mojibake(v) for k, v in row.items()} for row in reader]

    if rows:
        required_columns = {"question_no", "question"}
        missing = required_columns - set(rows[0].keys())
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    return rows


def repair_mojibake(text: Optional[str]) -> Optional[str]:
    """Fix UTF-8-decoded-as-a-single-byte-encoding-then-re-encoded text
    (e.g. "cafÃ©" instead of "café", "Le prophÃ¨te" instead of "Le prophète").

    questions_100.csv has this double-encoding on 10+ rows with accented
    names (French/Spanish/Czech/Norwegian/Romanian). It's not perfectly
    consistent: some bytes in the 0x80-0x9F range were mis-decoded as
    Windows-1252 (e.g. byte 0x83 -> "ƒ" the florin sign, as in "RÄƒchitova"
    for "Răchitova"), others passed through as raw Latin-1 control points
    (e.g. "Klokoč\x8dná" for "Klokočná"). Both repair strategies are tried
    in order; this is a pure in-memory fix when the file is loaded -
    questions_100.csv itself is never modified. Safe no-op on text that
    isn't actually double-encoded: each round-trip raises (and is caught)
    on already-correct UTF-8 text.
    """
    if text is None:
        return text
    for encoding in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
            if repaired != text:
                return repaired
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return text


def get_options(row: Dict[str, str]) -> Dict[str, str]:
    options = {}
    for letter in OPTION_LETTERS:
        value = row.get(letter, "")
        if value is not None and str(value).strip():
            options[letter] = str(value).strip()
    return options


# -----------------------------
# Text utilities
# -----------------------------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return [w for w in words if w not in GENERAL_STOPWORDS and len(w) > 1]


def extract_main_entity(question: str) -> str:
    """Heuristically pull the main proper-noun entity out of a question.

    Prefers quoted text (titles, names in quotes), otherwise picks the
    longest run of consecutive capitalised words that aren't question
    stopwords - this captures things like "P. Padmarajan" or
    "United States Marine Forces Special Operations Command".
    """
    text = question.strip().rstrip("?").strip()

    quote_match = re.search(r'["“]([^"”]{2,60})["”]', text)
    if quote_match:
        return quote_match.group(1).strip()

    words = text.split()
    candidates: List[List[str]] = []
    current: List[str] = []
    for word in words:
        bare = re.sub(r"[^\w.'-]", "", word)
        if not bare:
            if current:
                candidates.append(current)
                current = []
            continue
        if bare[0].isupper() and bare.lower() not in QUESTION_LEADING_STOPWORDS:
            current.append(bare)
        else:
            if current:
                candidates.append(current)
                current = []
    if current:
        candidates.append(current)

    if candidates:
        best = max(candidates, key=lambda c: len(" ".join(c)))
        return " ".join(best)
    return ""


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlsplit(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        return f"{netloc}{path}"
    except Exception:
        return url.lower()


def fingerprint_text(text: str) -> str:
    norm = re.sub(r"[^a-z0-9 ]", "", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm[:160]


# -----------------------------
# Query construction
# -----------------------------

def pick_distinctive_keywords(options: Dict[str, str], limit: int = 2) -> List[str]:
    """Find keywords that appear in exactly one option (i.e. distinguish it
    from the others), so we can search for the option's most specific term
    instead of dumping every option into one query."""
    token_counts: Counter = Counter()
    option_tokens: Dict[str, List[str]] = {}
    for letter, text in options.items():
        toks = tokenize(text)
        option_tokens[letter] = toks
        for t in set(toks):
            token_counts[t] += 1

    candidates = []
    for toks in option_tokens.values():
        distinctive = sorted({t for t in toks if token_counts[t] == 1}, key=len, reverse=True)
        if distinctive:
            candidates.append(distinctive[0])

    candidates = sorted(set(candidates), key=len, reverse=True)
    return candidates[:limit]


def build_search_queries(row: Dict[str, str]) -> List[str]:
    """Build 3-6 short, targeted search queries instead of one giant query
    containing the question plus every option (the starter code's weakness).
    """
    question = normalize_whitespace(str(row.get("question", "")))
    options = get_options(row)
    entity = extract_main_entity(question)

    queries: List[str] = []

    if question:
        queries.append(question.rstrip("?"))

    if entity:
        queries.append(f'"{entity}"')

        question_tokens = tokenize(question)
        entity_tokens = set(tokenize(entity))
        property_tokens = [t for t in question_tokens if t not in entity_tokens]
        if property_tokens:
            queries.append(f"{entity} {' '.join(property_tokens[:4])}")

        queries.append(f"{entity} wikipedia")

    base_for_option_query = entity if entity else question.rstrip("?")
    for term in pick_distinctive_keywords(options, limit=2):
        queries.append(f"{base_for_option_query} {term}")

    queries.append(question)

    deduped: List[str] = []
    seen = set()
    for q in queries:
        q_norm = normalize_whitespace(q)
        key = q_norm.lower()
        if not q_norm or key in seen:
            continue
        seen.add(key)
        deduped.append(q_norm)

    if len(deduped) < 3:
        for text in options.values():
            first_word = text.split()[0] if text.split() else ""
            q = normalize_whitespace(f"{base_for_option_query} {first_word}")
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                deduped.append(q)
            if len(deduped) >= 3:
                break

    return deduped[:MAX_QUERIES_PER_QUESTION]


# -----------------------------
# Evidence retrieval
# -----------------------------

def search_duckduckgo(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict[str, str]]:
    """Retrieve evidence snippets from DuckDuckGo for a single query.

    Never raises - a failed query just returns an empty list so one bad
    search doesn't crash the whole 100-question run.
    """
    global _ddgs_warned

    if not DDGS_AVAILABLE:
        if not _ddgs_warned:
            print("[Warning] 'ddgs' package not installed. No web evidence will be retrieved.")
            _ddgs_warned = True
        return []

    if not ENABLE_WEB_SEARCH:
        return []

    results_out = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for item in results:
                results_out.append(
                    {
                        "title": item.get("title", "") or "",
                        "snippet": item.get("body", "") or "",
                        "url": item.get("href", "") or "",
                        "query": query,
                    }
                )
    except Exception as error:
        print(f"[Warning] DuckDuckGo search failed for query '{query}': {error}")

    time.sleep(SEARCH_SLEEP_SECONDS)
    return results_out


def gather_evidence(queries: List[str]) -> List[Dict[str, str]]:
    """Run all queries for a question and collect raw (un-deduplicated) evidence."""
    all_evidence: List[Dict[str, str]] = []
    for query in queries:
        all_evidence.extend(search_duckduckgo(query))
    return all_evidence


def dedupe_evidence(evidence: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deduplicate by normalized URL and by a normalized title+snippet fingerprint
    (the same fact often shows up verbatim across multiple queries)."""
    seen_urls = set()
    seen_fingerprints = set()
    deduped = []
    for item in evidence:
        url_key = normalize_url(item.get("url", ""))
        fingerprint = fingerprint_text(f"{item.get('title', '')} {item.get('snippet', '')}")
        if url_key and url_key in seen_urls:
            continue
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        if url_key:
            seen_urls.add(url_key)
        if fingerprint:
            seen_fingerprints.add(fingerprint)
        deduped.append(item)
    return deduped


# -----------------------------
# Evidence ranking
# -----------------------------

def score_evidence_item(
    item: Dict[str, str],
    question_tokens: set,
    option_token_sets: Dict[str, set],
    entity_lower: str,
    exact_phrases: List[str],
) -> float:
    title = item.get("title", "") or ""
    snippet = item.get("snippet", "") or ""
    text = f"{title} {snippet}"
    text_lower = text.lower()
    tokens = set(tokenize(text))

    score = 0.0

    if question_tokens:
        overlap = len(tokens & question_tokens) / len(question_tokens)
        score += overlap * 2.0

    best_option_overlap = 0.0
    for opt_tokens in option_token_sets.values():
        if opt_tokens:
            overlap = len(tokens & opt_tokens) / len(opt_tokens)
            best_option_overlap = max(best_option_overlap, overlap)
    score += best_option_overlap * 1.5

    if entity_lower and entity_lower in text_lower:
        score += 1.5

    title_tokens = set(tokenize(title))
    if question_tokens:
        title_overlap = len(title_tokens & question_tokens) / len(question_tokens)
        score += title_overlap * 1.0

    for phrase in exact_phrases:
        phrase = phrase.strip().lower()
        if len(phrase) > 6 and phrase in text_lower:
            score += 1.0
            break

    url = (item.get("url", "") or "").lower()
    for domain, bonus in AUTHORITATIVE_DOMAINS.items():
        if domain in url:
            score += bonus
            break

    return score


def rank_evidence(
    row: Dict[str, str],
    evidence: List[Dict[str, str]],
    options: Dict[str, str],
    top_k: int = TOP_K_EVIDENCE,
) -> List[Dict[str, str]]:
    """Rank deduplicated evidence using deterministic keyword/phrase/source
    scoring (no embeddings needed - keeps the pipeline fast and explainable)."""
    question = str(row.get("question", ""))
    question_tokens = set(tokenize(question))
    option_token_sets = {letter: set(tokenize(text)) for letter, text in options.items()}
    entity = extract_main_entity(question)
    entity_lower = entity.lower()
    exact_phrases = list(options.values()) + ([entity] if entity else [])

    scored = []
    for item in evidence:
        relevance = score_evidence_item(item, question_tokens, option_token_sets, entity_lower, exact_phrases)
        item_copy = dict(item)
        item_copy["relevance_score"] = round(relevance, 4)
        scored.append(item_copy)

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:top_k]


# -----------------------------
# Option scoring
# -----------------------------

def extract_numbers(text: str) -> set:
    """Extract numeric substrings (including decimals) for robust fact matching.

    Wording around numbers varies a lot ("22.8-mile-long" vs "22.8 miles"),
    so matching the digits directly is far more reliable than word-token
    overlap for length/date/year/population-style answers.
    """
    return set(re.findall(r"\d+\.?\d*", text))


def score_options(
    options: Dict[str, str], ranked_evidence: List[Dict[str, str]], question: str = ""
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Score how strongly the ranked evidence supports each option.

    Returns (option_scores, exact_match_flags). Scores are normalised
    relative to the strongest option so the dict is easy to read/compare,
    e.g. {"A": 0.32, "B": 0.81, "C": 0.14, "D": 0.09, "E": 0.18}.
    """
    raw_scores = {letter: 0.0 for letter in options}
    exact_match = {letter: False for letter in options}

    # Only numeric tokens shared with the question (e.g. a route/highway
    # number repeated in every search result) are excluded from option
    # overlap - a bare identifier number spuriously inflates whichever
    # option happens to start with that same digit. Shared CONTENT WORDS
    # (e.g. "aircraft" in "...Liberty L-6 aircraft engine...") are kept:
    # those are often a legitimate, intentional hint baked into the question
    # itself, not noise, and stripping them would throw away a real signal.
    question_tokens = set(tokenize(question))
    shared_numeric_noise = {t for t in question_tokens if t.isdigit()}
    option_token_sets = {
        letter: set(tokenize(text)) - shared_numeric_noise for letter, text in options.items()
    }
    option_numbers = {letter: extract_numbers(text) for letter, text in options.items()}

    token_counts: Counter = Counter()
    for toks in option_token_sets.values():
        for t in toks:
            token_counts[t] += 1
    distinctive_sets = {
        letter: {t for t in toks if token_counts[t] == 1} for letter, toks in option_token_sets.items()
    }

    for rank, item in enumerate(ranked_evidence):
        weight = 1.0 / (rank + 1)
        title = item.get("title", "") or ""
        snippet = item.get("snippet", "") or ""
        raw_text = f"{title} {snippet}"
        text_lower = raw_text.lower()
        ev_tokens = set(tokenize(raw_text)) - shared_numeric_noise
        ev_numbers = extract_numbers(raw_text)
        title_lower = title.lower()

        for letter, opt_tokens in option_token_sets.items():
            if not opt_tokens:
                continue

            overlap = len(ev_tokens & opt_tokens) / len(opt_tokens)
            raw_scores[letter] += overlap * weight * 1.0

            # Lower weight than the base overlap fraction on purpose: a
            # "distinctive" single-word match (e.g. "fighter" matching only
            # because option D says "fighter jets") can be a false signal
            # when it shows up in unrelated evidence text (e.g. "fighter
            # aircraft", a WWI biplane, not a jet). The denominator-normalised
            # overlap fraction below is the more reliable primary signal.
            dist_overlap = len(ev_tokens & distinctive_sets[letter])
            raw_scores[letter] += dist_overlap * weight * 0.3

            option_text_lower = options[letter].lower()
            phrase_chunk = option_text_lower[:60]
            if len(phrase_chunk) > 8 and phrase_chunk in text_lower:
                raw_scores[letter] += 2.0 * weight
                exact_match[letter] = True

            if option_text_lower[:30] and option_text_lower[:30] in title_lower:
                raw_scores[letter] += 1.0 * weight

            numbers = option_numbers[letter]
            if numbers and numbers.issubset(ev_numbers):
                raw_scores[letter] += 2.5 * weight
                exact_match[letter] = True

    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score > 0:
        normalized = {letter: round(val / max_score, 4) for letter, val in raw_scores.items()}
    else:
        normalized = {letter: 0.0 for letter in raw_scores}

    return normalized, exact_match


def best_deterministic_answer(option_scores: Dict[str, float]) -> str:
    if not option_scores:
        return "Unknown"
    best_letter = max(option_scores, key=option_scores.get)
    if option_scores[best_letter] <= 0:
        return "Unknown"
    return best_letter


def compute_confidence(
    option_scores: Dict[str, float],
    ranked_evidence: List[Dict[str, str]],
    exact_match_flags: Dict[str, bool],
) -> float:
    """Combine top option score, gap to runner-up, evidence quantity, and
    exact phrase matches into a single 0-1 confidence value used to decide
    whether to trust the deterministic scorer over the LLM.

    Gap is weighted heavily (0.7) on purpose: a high top score that is TIED
    with another option means the scorer can't actually discriminate between
    them, so a tie gets an explicit penalty rather than being treated as a
    confident answer.
    """
    if not option_scores or max(option_scores.values()) <= 0:
        return 0.0

    sorted_scores = sorted(option_scores.values(), reverse=True)
    best = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    gap = best - second
    base = 0.3 * best + 0.7 * gap

    tie_count = sum(1 for v in option_scores.values() if abs(v - best) < 1e-6)
    if tie_count > 1:
        base *= 0.3

    evidence_bonus = min(0.15, 0.03 * len(ranked_evidence))
    exact_bonus = 0.15 if any(exact_match_flags.values()) else 0.0

    confidence = max(0.0, min(1.0, base + evidence_bonus + exact_bonus))
    return round(confidence, 4)


# -----------------------------
# RAG prompt
# -----------------------------

def build_prompt(
    row: Dict[str, str],
    options: Dict[str, str],
    ranked_evidence: List[Dict[str, str]],
    option_scores: Dict[str, float],
) -> str:
    """Build a RAG prompt where the LLM acts as an evidence judge, not a
    memory-based guesser."""
    lines = [
        "You are an evidence-based multiple-choice exam judge.",
        "Base your answer on the evidence below. If evidence is available, prioritise it",
        "over your own prior knowledge. If no evidence was retrieved, use your best general",
        "knowledge, but answer Unknown if you are not reasonably confident.",
        "",
        f"Question: {row.get('question', '')}",
        "",
        "Options:",
    ]
    for letter in OPTION_LETTERS:
        if letter in options:
            lines.append(f"{letter}. {options[letter]}")

    lines.append("")
    lines.append("Evidence (ranked by relevance):")
    if ranked_evidence:
        for i, item in enumerate(ranked_evidence, start=1):
            lines.append(f"[{i}] Title: {item.get('title', '')}")
            lines.append(f"    Snippet: {item.get('snippet', '')}")
            lines.append(f"    URL: {item.get('url', '')}")
    else:
        lines.append("(No evidence retrieved.)")

    lines.append("")
    lines.append("Deterministic evidence-overlap scores per option (higher = more textual support):")
    for letter in OPTION_LETTERS:
        if letter in option_scores:
            lines.append(f"  {letter}: {option_scores[letter]}")

    lines.append("")
    lines.append("Instructions: choose the option letter most strongly supported by the evidence above.")
    lines.append("Respond with EXACTLY ONE TOKEN: A, B, C, D, E, or Unknown. No explanation, no punctuation.")
    lines.append("Answer:")
    return "\n".join(lines)


# -----------------------------
# Local <=8B LLM (Ollama)
# -----------------------------

def get_model_name() -> str:
    return os.environ.get("HACKATHON_MODEL", DEFAULT_MODEL)


def call_ollama(prompt: str) -> Optional[str]:
    """Call a local <=8B model through Ollama. Returns None (never raises)
    if Ollama/the model is unavailable, so the pipeline can fall back to the
    deterministic scorer and still produce a valid CSV."""
    global _ollama_warned

    if DISABLE_LLM:
        return None

    if not REQUESTS_AVAILABLE:
        if not _ollama_warned:
            print("[Warning] 'requests' package not installed - skipping local LLM, using deterministic fallback only.")
            _ollama_warned = True
        return None

    model = get_model_name()
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 8},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as error:
        if not _ollama_warned:
            print(f"[Warning] Could not reach Ollama model '{model}' at {OLLAMA_HOST}: {error}")
            print("[Warning] Falling back to deterministic evidence-based scoring for all remaining questions.")
            _ollama_warned = True
        return None


# -----------------------------
# Answer validation
# -----------------------------

def clean_answer(answer) -> str:
    """Validate and normalise a raw model/deterministic answer.

    Accepts only A, B, C, D, E, or Unknown. Handles responses like
    "The answer is B." or "B" or stray punctuation. Never returns blank.
    """
    if answer is None:
        return "Unknown"

    text = str(answer).strip()
    if not text:
        return "Unknown"

    if text.strip().lower() == "unknown":
        return "Unknown"

    compact = re.sub(r"[^A-Za-z]", "", text)
    if len(compact) == 1 and compact.upper() in OPTION_LETTERS:
        return compact.upper()

    match = re.search(r"answer[^A-Za-z]{0,10}\b([A-E])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    if re.search(r"unknown", text, re.IGNORECASE):
        return "Unknown"

    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)

    return "Unknown"


def decide_final_answer(deterministic_answer: str, llm_answer: str, confidence: float) -> Tuple[str, str]:
    """Use confidence to decide whether to trust the deterministic evidence
    scorer or the LLM judge."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD and deterministic_answer != "Unknown":
        return deterministic_answer, "deterministic-high-confidence"
    if llm_answer and llm_answer != "Unknown":
        return llm_answer, "llm"
    if deterministic_answer != "Unknown":
        return deterministic_answer, "deterministic-fallback"
    return "Unknown", "unknown"


# -----------------------------
# Main per-question pipeline
# -----------------------------

def answer_question(row: Dict[str, str]) -> Dict:
    """Run the full evidence-grounded pipeline for one question."""
    options = get_options(row)
    queries = build_search_queries(row)

    raw_evidence = gather_evidence(queries)
    deduped_evidence = dedupe_evidence(raw_evidence)
    ranked = rank_evidence(row, deduped_evidence, options)

    option_scores, exact_flags = score_options(options, ranked, str(row.get("question", "")))
    deterministic_answer = best_deterministic_answer(option_scores)
    confidence = compute_confidence(option_scores, ranked, exact_flags)

    prompt = build_prompt(row, options, ranked, option_scores)
    llm_raw = call_ollama(prompt)
    llm_answer = clean_answer(llm_raw) if llm_raw is not None else "Unknown"

    final_answer, decision_reason = decide_final_answer(deterministic_answer, llm_answer, confidence)
    final_answer = clean_answer(final_answer)

    return {
        "question_no": row.get("question_no"),
        "question": row.get("question"),
        "answer": final_answer,
        "confidence": confidence,
        "llm_answer": llm_answer,
        "deterministic_answer": deterministic_answer,
        "decision_reason": decision_reason,
        "search_queries": queries,
        "option_scores": option_scores,
        "ranked_evidence": ranked,
        "raw_evidence_count": len(raw_evidence),
        "deduped_evidence_count": len(deduped_evidence),
    }


# -----------------------------
# Output / validation
# -----------------------------

def write_submission_csv(predictions: List[Dict], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_no", "answer"])
        for p in predictions:
            writer.writerow([p["question_no"], p["answer"]])


def validate_submission(predictions: List[Dict], limit: Optional[int]) -> None:
    """Ensure the final CSV is exactly question_no,answer with only valid,
    non-blank answers before saving."""
    problems = []
    for p in predictions:
        answer = p.get("answer")
        if answer is None or str(answer).strip() == "":
            problems.append(f"Blank answer for question_no={p.get('question_no')}")
        elif answer not in ALLOWED_ANSWERS:
            problems.append(f"Invalid answer '{answer}' for question_no={p.get('question_no')}")

    if problems:
        raise ValueError("Submission validation failed:\n" + "\n".join(problems))

    if limit is None and len(predictions) != 100:
        print(f"[Warning] Expected 100 rows but produced {len(predictions)}. Check questions_100.csv.")
    elif limit is not None:
        print(
            f"[Info] Produced {len(predictions)} row(s) because --limit={limit} was used. "
            "Run without --limit for the full 100-question submission."
        )

    print("Submission validation passed: columns are 'question_no,answer' and all answers are valid.")


def write_debug_logs(results: List[Dict], logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

    debug_path = logs_dir / "predictions_debug.csv"
    with open(debug_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question_no",
                "question",
                "final_answer",
                "confidence",
                "llm_answer",
                "deterministic_answer",
                "search_queries",
                "option_scores",
                "top_evidence_titles",
                "top_evidence_urls",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["question_no"],
                    r["question"],
                    r["answer"],
                    r["confidence"],
                    r["llm_answer"],
                    r["deterministic_answer"],
                    " | ".join(r["search_queries"]),
                    json.dumps(r["option_scores"]),
                    " | ".join(e.get("title", "") for e in r["ranked_evidence"]),
                    " | ".join(e.get("url", "") for e in r["ranked_evidence"]),
                ]
            )

    evidence_log_path = logs_dir / "evidence_log.jsonl"
    with open(evidence_log_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "question_no": r["question_no"],
                        "queries": r["search_queries"],
                        "raw_evidence_count": r["raw_evidence_count"],
                        "deduped_evidence_count": r["deduped_evidence_count"],
                        "ranked_evidence": r["ranked_evidence"],
                        "option_scores": r["option_scores"],
                        "confidence": r["confidence"],
                        "llm_answer": r["llm_answer"],
                        "deterministic_answer": r["deterministic_answer"],
                        "final_answer": r["answer"],
                        "decision_reason": r["decision_reason"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved debug logs to: {debug_path} and {evidence_log_path}")


# -----------------------------
# Pipeline entrypoint
# -----------------------------

def run_pipeline(questions_file: str, output_file: str, limit: Optional[int] = None) -> List[Dict]:
    questions = load_questions(questions_file)

    if limit is not None:
        questions = questions[:limit]

    model_note = "disabled (--no-llm)" if DISABLE_LLM else get_model_name()
    print(f"Answering {len(questions)} question(s) using model='{model_note}' (Ollama at {OLLAMA_HOST})...")

    results = []
    for row in tqdm(questions, desc="Answering questions"):
        results.append(answer_question(row))

    predictions = [{"question_no": r["question_no"], "answer": r["answer"]} for r in results]

    validate_submission(predictions, limit)
    write_submission_csv(predictions, output_file)
    print(f"\nSaved submission file to: {output_file}")

    write_debug_logs(results, LOGS_DIR)

    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VeriQuest AI - BCU AI Hackathon 2026 evidence-grounded MCQ pipeline")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE), help="Path to questions_100.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output CSV filename")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of questions to test")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Ollama model (must be <=8B parameters). Defaults to $HACKATHON_MODEL or llama3.1:8b.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable the local LLM judge and use only deterministic evidence scoring.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model:
        os.environ["HACKATHON_MODEL"] = args.model
    if args.no_llm:
        DISABLE_LLM = True
    run_pipeline(args.questions, args.output, args.limit)
