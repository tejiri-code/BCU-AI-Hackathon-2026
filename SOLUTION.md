# Team Apexmind — BCU AI Hackathon 2026 Solution

A fully local, reproducible Retrieval-Augmented Generation (RAG) pipeline that
answers the 100 multiple-choice questions.

## Model

| | |
|---|---|
| **Primary answering LLM** | `qwen2.5:7b` (Qwen2.5 7B Instruct) — 7B params |
| **Tie-breaker LLM** | `qwen2.5:3b` (3B params) — second opinion on disagreements |
| **Parameters** | both **within the ≤ 8B rule** |
| **Runtime** | [Ollama](https://ollama.com) (local, no cloud API) |
| **Retrieval / ranking** | Wikipedia API + BM25 (`rank-bm25`) — no neural model needed |

No closed-source or cloud LLM is used to produce answers, so the submission is
fully reproducible on a laptop.

## How it works

```
question + options
        │
        ▼
1. Query construction      multiple queries: the natural-language question, a
                           keyword-only query, AND option-aware queries built
                           from entities found in the answer options (the key
                           entity is often named only in the options)
        │
        ▼
2. Retrieval (Wikipedia)   MediaWiki API: search each query, collect candidate
                           titles round-robin (each query's best hit survives) →
                           intro extracts for all + full article text for the
                           top-3 hits. DuckDuckGo fallback if Wikipedia is thin.
        │
        ▼
3. Passage ranking         split into paragraphs (capped per article) → BM25 vs
                           (question+options) + title-match boost (favour the
                           article whose title echoes the question entity)
        │
        ▼
4. RAG prompt              question + options + top-7 passages (temperature 0)
        │
        ▼
5. LLM answer              qwen2.5:7b → single letter A–E
        │
        ▼
6. Tie-breaker             if the 7B answer disagrees with the evidence-overlap
                           heuristic, qwen2.5:3b votes; it overrides only when it
                           AND the heuristic agree against the 7B (2-vs-1)
        │
        ▼
7. Numeric post-pass       for number/unit questions, match each option's
                           (number, unit) pairs against the evidence; fix
                           "unit-swap" distractors (e.g. 22.8 mi vs 36.7 mi)
        │
        ▼
8. Validate & export       Apexmind_submission.csv  (question_no, answer)
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

# 2. Pull the models (~4.7 GB + ~1.9 GB)
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b

# 3. Run the full pipeline (writes Apexmind_submission.csv)
python3 src/run.py

# 4. Apply the numeric/unit disambiguation post-pass
cd src && python3 numeric_pass.py && cd ..

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
| `src/run.py` | Main pipeline (retrieval → rank → RAG → answer + tie-breaker → export) |
| `src/numeric_pass.py` | Numeric/unit disambiguation post-pass |
| `Apexmind_submission.csv` | **Final answer file** |
| `run_log.jsonl` | Per-question log (answer, source, primary/tie-break/heuristic votes) |
| `numeric_corrections.log` | Overrides applied by the numeric post-pass |
| `slides/make_slides.py` | Builds the 3-slide PPTX presentation |
| `slides/Apexmind_presentation.pptx` | Presentation |
