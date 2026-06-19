# Team Apexmind — BCU AI Hackathon 2026 Solution

A fully local, reproducible Retrieval-Augmented Generation (RAG) pipeline that
answers the 100 multiple-choice questions.

## Model

| | |
|---|---|
| **Answering LLM** | `qwen2.5:7b` (Qwen2.5 7B Instruct) |
| **Parameters** | 7 billion — **within the ≤ 8B rule** |
| **Runtime** | [Ollama](https://ollama.com) (local, no cloud API) |
| **Embeddings / ranking** | BM25 (`rank-bm25`) — no neural model needed |

No closed-source or cloud LLM is used to produce answers, so the submission is
fully reproducible on a laptop.

## How it works

```
question + options
        │
        ▼
1. Query construction      natural-language question  +  keyword-only query
        │
        ▼
2. Retrieval (Wikipedia)   MediaWiki API: search → intro extracts for top hits
                           + full article text for the 2 best-matching articles
                           (DuckDuckGo fallback if Wikipedia returns too little)
        │
        ▼
3. Passage ranking         split into paragraphs → BM25 vs (question+options)
                           + title-match boost (favour the article whose title
                           echoes the question entity)
        │
        ▼
4. RAG prompt              question + options + top-7 passages
        │
        ▼
5. LLM answer              qwen2.5:7b → single letter A–E (temperature 0)
        │
        ▼
6. Fallback                if the model abstains, pick the option whose words
                           best overlap the retrieved evidence
        │
        ▼
7. Validate & export       Apexmind_submission.csv  (question_no, answer)
```

**Why Wikipedia-first?** The question set is sourced from Wikipedia articles, so
the MediaWiki API gives clean, on-topic passages with no scraper rate-limit
issues. Calls are throttled with exponential backoff on HTTP 429.

## How to run

Requirements: Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Install Python deps
python3 -m pip install -r starter_code/requirements.txt
python3 -m pip install ddgs rank-bm25

# 2. Pull the model (~4.7 GB)
ollama pull qwen2.5:7b

# 3. Run the full pipeline (writes Apexmind_submission.csv)
python3 src/run.py

# Optional: quick test on the first 5 questions
python3 src/run.py --limit 5
```

Outputs:
- `Apexmind_submission.csv` — final answers (100 rows, `question_no,answer`)
- `evidence_cache.json` — cached retrieved evidence (re-runs skip the network)
- `run_log.jsonl` — per-question log: chosen answer, source (llm/fallback),
  raw model output, and the top evidence passage (for inspection/debugging)

## Reproducibility notes

- LLM decoding is deterministic (`temperature = 0`).
- Retrieved evidence is cached, so a second run reproduces the same inputs to
  the model without hitting the network.
- All answers are validated to be one of `A`, `B`, `C`, `D`, `E`, or `Unknown`
  before export.

## Files

| Path | Purpose |
|---|---|
| `src/run.py` | The complete pipeline (retrieval → rank → RAG → answer → export) |
| `Apexmind_submission.csv` | Final answer file |
| `run_log.jsonl` | Per-question reasoning log |
| `slides/` | Presentation |
