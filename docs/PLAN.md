# ApexMind — BCU AI Hackathon 2026 QA Pipeline

## Context
We must answer 100 multiple-choice questions (`questions_100.csv`, options A–E) and emit
`ApexMind_submission.csv` (`question_no,answer`; values A–E or `Unknown`; exactly 100 rows),
maximising accuracy with a **reproducible, well-documented** pipeline using an **LLM ≤ 8B params**.
Internet-assisted answering is allowed.

**Key accuracy insight:** the questions are phrased *"according to Wikipedia"* (e.g. MARSOC capabilities,
P. Padmarajan, U.S. Route 11 in Georgia). So **Wikipedia-grounded RAG + cross-encoder reranking** is the
single biggest lever, on top of a strong ≤8B reasoner with self-consistency and a multi-model vote.

Everything runs locally via prebuilt **llama.cpp server** (OpenAI-compatible API) at
`C:\Users\nwagb\Desktop\thesis\tools\llamacpp\bin\llama-server.exe` on the RTX 4060 (8 GB VRAM).
All chosen models are GGUF Q4/Q8 in `models/` and are ≤8B (documented & SHA-recorded for compliance).

Decisions locked: team **ApexMind**; **max-accuracy ensemble**; **agreement + ~20-question manual gold set**
for evaluation; build **GitHub-ready locally, no remote push** until explicitly approved.

## Models (all ≤8B, in `models/`)
- **Primary reasoner:** `Qwen3-8B-Q4_K_M` (8B, at the cap — strongest knowledge).
- **Ensemble voters:** `Qwen3-4B-Instruct-2507-Q4_K_M`, `Qwen3.5-4B-Q4_K_M` (independent votes on hard Qs).
- **Embeddings:** `Qwen3-Embedding-0.6B-f16`. **Reranker:** `bge-reranker-v2-m3-Q8_0` (llama.cpp `/rerank`).
- Fallbacks present: `gemma-4-E4B`, `Ministral-3-3B`. Vision models kept but unused (questions are text-only).

## VRAM staging (8 GB → one LLM loaded at a time)
Pipeline runs in stages so each model loads exactly once (no mid-loop swaps):
1. **Retrieve** all 100 Qs (CPU/network) → cache raw evidence.
2. Start small **embed + rerank** servers (~1.5 GB) → rank passages for all 100 → cache top-k.
3. Start **Qwen3-8B** → answer all 100 (deterministic pass + self-consistency) → cache.
4. Low-confidence subset only: start **Qwen3-4B-2507**, then **Qwen3.5-4B** → vote.

## Repo structure (src layout, GitHub-ready)
```
README.md                 # full architecture, diagram, run steps, model+8B justification, results
pyproject.toml / requirements.txt   # pinned deps (Python 3.11 venv at .venv)
.gitignore                # .venv, models/, cache/, outputs/runs, __pycache__
LICENSE (MIT)
config/default.yaml       # paths, model registry, seeds, top_k, n_samples, ports, thresholds
src/apexmind/
  config.py               # YAML -> dataclass, path resolution
  models.py               # registry + size + sha256 + assert <=8B
  server.py               # launch/stop llama-server (subprocess), health check, context manager
  llm_client.py           # OpenAI-compatible chat/embeddings/rerank; retries (tenacity); seed/temp
  data.py                 # load+validate questions; write/validate submission (A-E, 100 rows)
  parsing.py              # robust final-letter extraction
  prompting.py            # MCQ templates: rag + closed-book (method diversity)
  retrieval/{query,wikipedia,web,chunk,rerank,evidence}.py   # retrieve->chunk->rerank->top-k (+cache)
  solver.py               # single-model answer + self-consistency majority vote
  ensemble.py             # multi-model vote, confidence, low-confidence escalation
  pipeline.py             # end-to-end orchestration, caching, run manifest (git sha, params)
  evaluate.py             # agreement matrix, gold-set accuracy, ablations, confidence report
  cli.py                  # commands: serve | retrieve | answer | ensemble | evaluate | all
scripts/{fetch_models.py(done), run_all.ps1, run_all.sh}
data/gold_dev.csv         # ~20 hand-labeled (verified via Wikipedia) for accuracy estimate
tests/{test_parsing,test_data,test_prompting}.py
outputs/ApexMind_submission.csv ; outputs/runs/<ts>/{manifest.json,confidence_report.csv,per_q_logs}
docs/architecture.md ; slides/ApexMind_presentation.pptx (python-pptx from template)
cache/                    # gitignored retrieval + LLM response caches
```

## Pipeline detail (per question)
1. **Query construction** (`retrieval/query.py`): build 1–3 queries from question + key option terms; entity-focused.
2. **Retrieval** (`wikipedia.py` via MediaWiki API + `web.py` via `ddgs`): cached JSON keyed by query hash.
3. **Rerank** (`rerank.py`): chunk evidence into passages, score with bge-reranker cross-encoder, keep top-k (≈5).
   (Qwen3-Embedding used for optional dense pre-filter / fallback when reranker server is down.)
4. **Prompt** (`prompting.py`): question + options A–E + top evidence; concise reasoning then `Answer: <letter>`.
   Two methods: evidence (RAG) and closed-book, for agreement-based confidence.
5. **Generate** (`solver.py`): Qwen3-8B temp-0 deterministic pass + N seeded samples (self-consistency vote).
6. **Parse** (`parsing.py`): regex the final letter; always commit a best guess (never `Unknown` unless unparseable).
7. **Confidence**: agreement across samples, methods (RAG vs closed-book), and models.
8. **Escalate** (`ensemble.py`): low-confidence Qs get extra retrieval + Qwen3-4B-2507 & Qwen3.5-4B votes; weighted majority.

## Reproducibility
Pinned deps, fixed seeds, temp-0 primary pass, model registry with SHA256 + 8B assertion, YAML config,
on-disk caches for retrieval and LLM responses, per-run `manifest.json` (timestamp, git commit, model files,
params), and one-command runners (`run_all.ps1` / `run_all.sh`).

## Evaluation (`evaluate.py`)
- Inter-model & inter-method **agreement matrix** + **confidence histogram** + coverage over all 100.
- **Accuracy on `data/gold_dev.csv`** (~20 questions I hand-verify against Wikipedia during the build).
- **Ablation**: closed-book vs RAG vs full ensemble on the gold set, to prove each component helps.
- `pytest` unit tests for parsing, data IO, prompt building.

## Build order
1. Scaffold repo (package, `config/default.yaml`, `.gitignore`, `pyproject.toml`, pinned `requirements.txt`, LICENSE).
2. Install deps into `.venv` (pandas, openpyxl, ddgs, openai/httpx, pyyaml, tenacity, rank-bm25, python-pptx, pytest, tqdm).
3. `models.py` (+ SHA256 manifest & ≤8B assertion); `server.py` + `llm_client.py`; smoke-test Qwen3-8B chat, embed, `/rerank`.
4. `data.py` + `parsing.py` + their tests.
5. Retrieval package + cache; smoke-test on 3 questions.
6. `prompting.py` + `solver.py`; test on 5 questions end-to-end with Qwen3-8B.
7. `ensemble.py` + `pipeline.py` + run manifest.
8. Run retrieval (100) → primary answering (100) → escalation ensemble on low-confidence.
9. Hand-label `data/gold_dev.csv` (~20) ; `evaluate.py` agreement + accuracy + ablations.
10. Emit & validate `outputs/ApexMind_submission.csv` (100 rows, A–E).
11. README (architecture + diagram + results + 8B justification) + `docs/architecture.md` + slides.
12. Run `pytest`; final end-to-end verification.

## Verification
- `pytest -q` green.
- `python -m apexmind.cli all --limit 5` runs clean end-to-end, then full 100.
- Submission validator confirms 100 rows, exact columns, all values in {A,B,C,D,E}.
- Evaluation report prints agreement/confidence and gold-set accuracy; ablation shows RAG+ensemble ≥ baselines.
- Models verified ≤8B (assertion + recorded sizes/SHAs) for documented compliance.

## Notes / risks
- Hybrid Qwen3-8B: default to non-thinking (`/no_think`) for deterministic speed; thinking mode available as a tunable for hard Qs.
- `ddgs` can rate-limit; caching + Wikipedia API primary mitigate this. Retrieval cached so reruns are cheap.
- No official key exists → reported accuracy is an estimate from the manual gold set; agreement is the all-100 proxy.
- No GitHub remote will be created/pushed until you explicitly approve.
