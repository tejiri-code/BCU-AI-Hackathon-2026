# ApexMind — BCU AI Hackathon 2026

A reproducible, Wikipedia-grounded **RAG + confidence** pipeline that answers 100 multiple-choice
questions using only **open models ≤ 8B parameters**, served locally via llama.cpp.

> **Final answer file:** [`ApexMind_submission.csv`](ApexMind_submission.csv) — 100 rows, `question_no,answer`.
> **One-notebook walkthrough:** [`src/run.ipynb`](src/run.ipynb).
> **Slides:** [`slides/ApexMind-BCU-AI-Hackathon-2026.pptx`](slides/ApexMind-BCU-AI-Hackathon-2026.pptx).

---

## Results

| Evaluation (20-question hand-verified gold set) | Accuracy |
|---|---|
| Closed-book (model only) | 90% |
| **RAG (Wikipedia evidence)** | **100%** |
| RAG + confidence (high-confidence subset) | 100% |

- **Retrieval adds +10 points** over the model alone.
- The pipeline is **byte-for-byte reproducible** (greedy decoding, temp 0, fixed seed, cached retrieval).
- Confidence scoring **correctly flags borderline questions** (low agreement / low support), which we route to sourced human verification.

## Models & 8B compliance

All models are GGUF and **≤ 8B parameters**, run on an RTX 4060 (8 GB) through the prebuilt
**llama.cpp** OpenAI-compatible server.

| Role | Model | Params |
|---|---|---|
| Primary reasoner | **Qwen3-8B** (Q4_K_M) | 8B (at cap) |
| Ensemble voters | Qwen3-4B-Instruct-2507, Qwen3.5-4B | 4B |
| Embeddings (RAG) | Qwen3-Embedding-0.6B | 0.6B |
| Reranker | bge-reranker-v2-m3 | 0.6B |

No model exceeds 8B. Model files + sizes are recorded in each run's `manifest.json`.

## Architecture

```
questions_100.csv
  → [1] query construction (entity-focused)
  → [2] retrieval (cached): Wikipedia MediaWiki API (primary) + DuckDuckGo (web)
  → [3] evidence assembly (top passages within a char budget; optional cross-encoder rerank)
  → [4] prompt: RAG + closed-book, instruct "Answer: X"
  → [5] Qwen3-8B (temp 0, /no_think) — deterministic
  → [6] robust parse → letter A–E
  → [7] confidence = parse + evidence + support + RAG/closed-book agreement
  → [8] sourced human verification of low-confidence / disputed items
  → ApexMind_submission.csv
```

Key insight: the questions are phrased *"according to Wikipedia"*, so Wikipedia-grounded retrieval
is the single biggest accuracy lever.

## Repository layout

```
src/run.ipynb     # single comprehensive, reproducible notebook
ApexMind_submission.csv   # FINAL answers (submission)
src/apexmind/               # reusable engine (config, retrieval, solver, scoring, pipeline, evaluate, cli)
config/default.yaml         # paths, model registry, generation/retrieval/eval settings
data/gold_dev.csv           # hand-verified gold set (with source URLs)
scripts/                    # fetch_models, seed_gold, build_final_submission, build_notebook, compare_risky
tests/                      # unit tests (parsing, data, scoring)
outputs/runs/               # per-run manifests, confidence reports, verification logs
docs/PLAN.md                # architecture plan;  HACKATHON_CONTEXT.md / MEMORY_HANDOFF.md
```

## Run it

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate           # Windows  (source .venv/bin/activate on *nix)
pip install -e .

python scripts/fetch_models.py        # download the ≤8B models (GGUF)
python -m apexmind.cli run            # full RAG pipeline → submission + confidence_report
python -m apexmind.cli evaluate --ablation   # gold-set accuracy + ablation
pytest -q                             # unit tests
```

Regenerate the final, human-verified submission deterministically:

```bash
python scripts/build_final_submission.py     # writes ApexMind_submission.csv
```

## How the final answers were produced

1. The Qwen3-8B RAG pipeline produced a baseline answer + a confidence score for all 100 questions.
2. Low-confidence and disputed questions were **verified against primary Wikipedia sources**
   (logged in [`outputs/runs/final_manual_verification.csv`](outputs/runs/final_manual_verification.csv)
   with a source URL per row, and a deep-research report).
3. **Final key = team deep-research key, overridden by the *sourced* manual verification** on the
   5 questions where they conflicted. All overrides are listed (with sources) in
   `scripts/build_final_submission.py` and are reversible with a one-line edit:

   | Q | Final | Source |
   |---|---|---|
   | 16 | E | The Other Bank (2009) → Giorgi Ovashvili |
   | 36 | B | NJ Turnpike mainline 117.20 mi (closest valid option) |
   | 65 | A | Tim Kinsella — filmmaking is the sourced profession |
   | 79 | **Unknown** | Răchitova — flawed item: all five options are Răchitova villages |
   | 88 | E | Davidkhanian Mansion — owned by the Iranian government |
   | 95 | E | Death (2000 Thy Serpent EP) |

`Unknown` is a permitted answer and is used only for the genuinely unanswerable Q79.

## Reproducibility

Temp 0 + fixed seed (deterministic greedy decoding), on-disk caches for retrieval, a per-run
`manifest.json` (timestamp, model file, params, settings), pinned dependencies, and unit tests.
Two full runs of the pipeline are byte-identical.

## Team

**ApexMind** — BCU AI Hackathon 2026.
