"""
BCU AI Hackathon 2026 — Team Apexmind
Generative-AI multiple-choice QA pipeline.

Pipeline:
    1. Load questions_100.csv.
    2. For each question, retrieve evidence from Wikipedia (primary) and
       DuckDuckGo (fallback). The question set is Wikipedia-sourced, so the
       MediaWiki API gives clean, reliable passages without scraper rate limits.
    3. Split evidence into passages and rank them with BM25 against the
       question + all options.
    4. Build a RAG prompt and ask a local <=8B LLM (qwen2.5:7b via Ollama) to
       return a single letter A-E.
    5. If the model abstains or fails, fall back to the option whose text best
       matches the retrieved evidence (BM25 overlap).
    6. Validate and export TEAM_submission.csv.

Model: qwen2.5:7b (7B parameters) served locally by Ollama. <= 8B rule satisfied.
No closed-source / cloud LLM is used for answering.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from rank_bm25 import BM25Okapi

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_QUESTIONS_FILE = ROOT_DIR / "questions_100.csv"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "Apexmind_submission.csv"
CACHE_FILE = ROOT_DIR / "evidence_cache.json"
LOG_FILE = ROOT_DIR / "run_log.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
MODEL_NAME = "qwen2.5:7b"          # primary answering model (<= 8B)
TIEBREAK_MODEL = "qwen2.5:3b"      # cheap second opinion on disagreements (<= 8B)
ENABLE_TIEBREAK = True

# Hugging Face ids used when Ollama is not available (e.g. on Colab). The
# backend is auto-detected at runtime, so the same code runs locally (Ollama)
# or on a Colab GPU (transformers).
HF_MODEL_MAP = {
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
}
LLM_BACKEND = None                 # resolved lazily: "ollama" or "transformers"

OPTIONS = ["A", "B", "C", "D", "E"]
ALLOWED_ANSWERS = {"A", "B", "C", "D", "E", "Unknown"}

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SEARCH_LIMIT = 4          # candidate articles per question
TOP_PASSAGES = 7               # passages fed to the LLM
USER_AGENT = "BCU-Hackathon-Apexmind/1.0 (educational use)"

WIKI_MIN_INTERVAL = 0.4        # polite gap between Wikipedia API calls (s)

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": USER_AGENT})

_last_wiki_call = [0.0]


def wiki_get(params: dict, retries: int = 3) -> dict:
    """GET the MediaWiki API with throttling + exponential backoff on 429."""
    for attempt in range(retries + 1):
        wait = WIKI_MIN_INTERVAL - (time.time() - _last_wiki_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_wiki_call[0] = time.time()
        r = HTTP.get(WIKI_API, params=params, timeout=20)
        if r.status_code == 429:
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    return {}


# -----------------------------
# Data loading
# -----------------------------

def load_questions(path: Path) -> pd.DataFrame:
    questions = pd.read_csv(path)
    required = {"question_no", "question"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return questions


# -----------------------------
# Text helpers
# -----------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "when", "where", "how", "why", "did",
    "does", "do", "with", "by", "as", "at", "from", "that", "this", "it", "its",
    "his", "her", "their", "according", "wikipedia", "excerpt", "provided",
    "mentioned", "best", "known", "primary", "main", "one", "following",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def keywords(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) > 1]


def option_texts(row: pd.Series) -> Dict[str, str]:
    out = {}
    for opt in OPTIONS:
        val = row.get(opt, "")
        if pd.notna(val) and str(val).strip():
            out[opt] = str(val).strip()
    return out


# -----------------------------
# Evidence retrieval
# -----------------------------

def wiki_search_titles(query: str, limit: int = WIKI_SEARCH_LIMIT) -> List[str]:
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": limit, "format": "json", "srprop": "",
    }
    try:
        data = wiki_get(params)
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]
    except Exception as e:
        print(f"    [wiki search failed] {e}")
        return []


def wiki_extracts(titles: List[str]) -> List[Dict[str, str]]:
    """Fetch plain-text intro extracts for a batch of article titles."""
    if not titles:
        return []
    params = {
        "action": "query", "prop": "extracts", "exintro": 1,
        "explaintext": 1, "exlimit": "max", "format": "json", "redirects": 1,
        "titles": "|".join(titles),
    }
    try:
        pages = wiki_get(params).get("query", {}).get("pages", {})
        out = []
        for page in pages.values():
            text = page.get("extract", "")
            if text:
                out.append({"title": page.get("title", ""), "snippet": text,
                            "url": f"https://en.wikipedia.org/wiki/{page.get('title','').replace(' ', '_')}"})
        return out
    except Exception as e:
        print(f"    [wiki extract failed] {e}")
        return []


def wiki_full_extract(title: str, max_chars: int = 6000) -> Optional[Dict[str, str]]:
    """Fetch the full plain-text body of a single article (answers often sit
    deeper than the intro, e.g. an infobox-style fact)."""
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "exlimit": 1, "format": "json", "redirects": 1, "titles": title,
    }
    try:
        pages = wiki_get(params).get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            if text:
                return {"title": page.get("title", ""), "snippet": text[:max_chars],
                        "url": f"https://en.wikipedia.org/wiki/{page.get('title','').replace(' ', '_')}"}
    except Exception as e:
        print(f"    [wiki full extract failed] {e}")
    return None


def ddg_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not DDGS_AVAILABLE:
        return []
    out = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                out.append({"title": item.get("title", ""),
                            "snippet": item.get("body", ""),
                            "url": item.get("href", "")})
    except Exception as e:
        print(f"    [ddg failed] {e}")
    return out


_CAP_RE = re.compile(r"\b[A-Z][\w'’.&-]*(?:\s+[A-Z][\w'’.&-]*)*\b")


def proper_nouns(text: str) -> List[str]:
    """Capitalised multi-word phrases (rough named-entity proxy)."""
    return [m for m in _CAP_RE.findall(text or "") if len(m) > 2]


def shared_option_terms(opts: Dict[str, str]) -> List[str]:
    """Content words that appear in >=2 options — usually the subject named in
    the options (e.g. 'Raw Melody Men'), which is often missing from the
    question itself."""
    from collections import Counter
    c = Counter()
    for v in opts.values():
        for w in set(keywords(v)):
            c[w] += 1
    return [w for w, n in c.most_common() if n >= 2]


def build_queries(row: pd.Series) -> List[str]:
    """Build a small ordered set of search queries. Crucially, several queries
    pull in entities from the OPTIONS, not just the question — many answers name
    their key entity only in the option text."""
    question = str(row.get("question", ""))
    opts = option_texts(row)
    qkw = keywords(question)

    queries = [question, " ".join(qkw[:8])]

    shared = shared_option_terms(opts)
    if shared:
        queries.append(" ".join(qkw[:5] + shared[:6]))

    # The single most entity-rich option (its proper nouns + question keywords).
    if opts:
        best = max(opts.values(), key=lambda v: len(proper_nouns(v)))
        pn = proper_nouns(best)
        if pn:
            queries.append(" ".join(qkw[:4] + pn[:3]))

    out, seen = [], set()
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:4]


def retrieve_evidence(row: pd.Series) -> List[Dict[str, str]]:
    """Gather evidence documents for a question from Wikipedia + DDG fallback."""
    question = str(row.get("question", ""))
    opts = option_texts(row)

    # Collect candidate titles round-robin across queries so that EACH query's
    # best hit survives the cap (the option-aware queries often pinpoint the
    # exact answer article, and must not be crowded out by earlier queries).
    per_query = [wiki_search_titles(q) for q in build_queries(row)]
    titles: List[str] = []
    for rank in range(max((len(p) for p in per_query), default=0)):
        for hits in per_query:
            if rank < len(hits) and hits[rank] not in titles:
                titles.append(hits[rank])
    titles = titles[:8]

    # Intro extracts for all candidates + full body for the top 3 hits
    # (many factual answers live below the intro).
    evidence = wiki_extracts(titles)
    for t in titles[:3]:
        full = wiki_full_extract(t)
        if full:
            evidence.append(full)

    # Fallback to web search if Wikipedia returned little.
    if len(" ".join(e["snippet"] for e in evidence)) < 200:
        evidence += ddg_search(question + " " + " ".join(opts.values()))

    return evidence


# -----------------------------
# Passage ranking (BM25)
# -----------------------------

def to_passages(evidence: List[Dict[str, str]], max_per_title: int = 4) -> List[Dict[str, str]]:
    """Split evidence docs into paragraph passages, keeping their source title.
    Caps passages per article so one large/generic article cannot flood the
    ranker and crowd out the right (more specific) source."""
    passages = []
    per_title: Dict[str, int] = {}
    for e in evidence:
        title = e.get("title", "")
        for para in re.split(r"\n+", e.get("snippet", "")):
            para = para.strip()
            if len(para) > 40 and per_title.get(title, 0) < max_per_title:
                passages.append({"title": title, "text": para})
                per_title[title] = per_title.get(title, 0) + 1
    return passages


def rank_passages(row: pd.Series, passages: List[Dict[str, str]], top_k: int = TOP_PASSAGES) -> List[str]:
    """BM25 over passage text + a bonus for passages whose article TITLE matches
    the question entity (the right article's title usually echoes the question)."""
    if not passages:
        return []
    q_kw = keywords(str(row.get("question", "")))
    query = q_kw + [w for v in option_texts(row).values() for w in keywords(v)]
    if not query:
        return [f"{p['title']}: {p['text']}" for p in passages[:top_k]]

    docs = [keywords(f"{p['title']} {p['text']}") for p in passages]
    bm25 = BM25Okapi(docs)
    bm_scores = bm25.get_scores(query)

    q_set = set(q_kw)
    ranked = []
    for p, bm in zip(passages, bm_scores):
        title_kw = set(keywords(p["title"]))
        title_overlap = len(title_kw & q_set) / (len(q_set) or 1)
        ranked.append((bm + 4.0 * title_overlap, f"{p['title']}: {p['text']}"))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in ranked[:top_k]]


def evidence_overlap_guess(row: pd.Series, passages: List[str]) -> str:
    """Heuristic fallback: option whose distinctive words best match evidence."""
    opts = option_texts(row)
    if not opts or not passages:
        return "Unknown"
    ev_tokens = set(w for p in passages for w in keywords(p))
    best, best_score = "Unknown", -1.0
    for opt, text in opts.items():
        kw = keywords(text)
        if not kw:
            continue
        score = sum(1 for w in kw if w in ev_tokens) / len(kw)
        if score > best_score:
            best, best_score = opt, score
    return best


# -----------------------------
# Prompt + LLM answering
# -----------------------------

def build_prompt(row: pd.Series, passages: List[str]) -> str:
    question = str(row.get("question", ""))
    opts = option_texts(row)
    options_block = "\n".join(f"{k}. {v}" for k, v in opts.items())
    evidence_block = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages)) or "(no evidence retrieved)"
    return f"""You are answering a multiple-choice general-knowledge question.
Use the EVIDENCE when it is relevant; otherwise rely on your own knowledge.
Match exact numbers and units carefully (e.g. miles vs kilometres).
Think step by step internally, then output ONLY the single letter of the best option.

EVIDENCE:
{evidence_block}

QUESTION:
{question}

OPTIONS:
{options_block}

Respond with exactly one character: the letter (A, B, C, D, or E) of the correct option. No explanation."""


def detect_backend() -> str:
    """Use a local Ollama server if reachable, else fall back to Hugging Face
    transformers (auto-detected once and cached)."""
    global LLM_BACKEND
    if LLM_BACKEND is None:
        try:
            HTTP.get(OLLAMA_TAGS, timeout=3).raise_for_status()
            LLM_BACKEND = "ollama"
        except Exception:
            LLM_BACKEND = "transformers"
        print(f"[LLM backend: {LLM_BACKEND}]")
    return LLM_BACKEND


_HF_PIPES: Dict[str, object] = {}


def _hf_generate(prompt: str, model: str) -> Optional[str]:
    """Generate with a local transformers model (lazy-loaded, cached)."""
    try:
        if model not in _HF_PIPES:
            import torch
            from transformers import pipeline
            hf_id = HF_MODEL_MAP.get(model, model)
            _HF_PIPES[model] = pipeline(
                "text-generation", model=hf_id,
                torch_dtype="auto", device_map="auto",
            )
        pipe = _HF_PIPES[model]
        out = pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=8, do_sample=False,
            pad_token_id=getattr(pipe.tokenizer, "eos_token_id", None),
        )
        return out[0]["generated_text"][-1]["content"]
    except Exception as e:
        print(f"    [transformers failed] {e}")
        return None


def _ollama_generate(prompt: str, model: str, retries: int = 2) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }
    for attempt in range(retries + 1):
        try:
            r = HTTP.post(OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except Exception as e:
            if attempt == retries:
                print(f"    [ollama failed] {e}")
                return None
            time.sleep(1.5)
    return None


def ask_llm(prompt: str, model: str = MODEL_NAME, retries: int = 2) -> Optional[str]:
    """Backend-agnostic single-letter answer. Uses Ollama locally, transformers
    on machines without Ollama (e.g. Colab)."""
    if detect_backend() == "ollama":
        return _ollama_generate(prompt, model, retries)
    return _hf_generate(prompt, model)


def parse_letter(text: Optional[str], valid: List[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"[A-Ea-e]", text)
    if m:
        letter = m.group(0).upper()
        if letter in valid:
            return letter
    return None


# -----------------------------
# Caching
# -----------------------------

def load_cache() -> Dict[str, List[Dict[str, str]]]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache))


# -----------------------------
# Pipeline
# -----------------------------

def answer_question(row: pd.Series, cache: Dict) -> Dict:
    qno = str(row.get("question_no"))
    opts = option_texts(row)
    valid = list(opts.keys())

    if qno in cache:
        evidence = cache[qno]
    else:
        evidence = retrieve_evidence(row)
        cache[qno] = evidence
        save_cache(cache)

    passages = rank_passages(row, to_passages(evidence))
    prompt = build_prompt(row, passages)

    raw = ask_llm(prompt, MODEL_NAME)
    primary = parse_letter(raw, valid)
    heuristic = evidence_overlap_guess(row, passages)

    tiebreak = None
    if primary is None:
        # Model abstained / unparseable -> heuristic fallback.
        answer, source = heuristic, "fallback"
    elif not ENABLE_TIEBREAK or primary == heuristic:
        answer, source = primary, "llm"
    else:
        # Primary disagrees with the evidence-overlap signal: get a cheap second
        # opinion from the 3B model and take a majority vote. The 3B only flips
        # the answer when it AND the heuristic agree against the 7B (2 vs 1);
        # otherwise the stronger 7B wins ties.
        tb_raw = ask_llm(prompt, TIEBREAK_MODEL)
        tiebreak = parse_letter(tb_raw, valid)
        if tiebreak and tiebreak == heuristic and tiebreak != primary:
            answer, source = tiebreak, "tiebreak"
        else:
            answer, source = primary, "llm"

    with LOG_FILE.open("a") as f:
        f.write(json.dumps({
            "question_no": qno, "answer": answer, "source": source,
            "raw": raw, "primary": primary, "tiebreak": tiebreak,
            "heuristic": heuristic, "n_passages": len(passages),
            "top_passage": passages[0] if passages else "",
        }) + "\n")

    return {"question_no": row.get("question_no"), "answer": answer}


def run_pipeline(questions_file: Path, output_file: Path, limit: Optional[int] = None) -> pd.DataFrame:
    questions = load_questions(questions_file)
    if limit is not None:
        questions = questions.head(limit)

    cache = load_cache()
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    predictions = []
    total = len(questions)
    for i, (_, row) in enumerate(questions.iterrows(), 1):
        t0 = time.time()
        result = answer_question(row, cache)
        print(f"[{i}/{total}] Q{result['question_no']} -> {result['answer']}  ({time.time()-t0:.1f}s)")
        predictions.append(result)

    submission = pd.DataFrame(predictions, columns=["question_no", "answer"])
    # Final validation
    submission["answer"] = submission["answer"].apply(
        lambda a: a if a in ALLOWED_ANSWERS else "Unknown"
    )
    submission.to_csv(output_file, index=False)
    print(f"\nSaved {len(submission)} answers to: {output_file}")
    return submission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BCU AI Hackathon 2026 — Apexmind")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_FILE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.questions, args.output, args.limit)
