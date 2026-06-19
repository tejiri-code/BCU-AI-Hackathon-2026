# VeriQuest AI — BCU AI Hackathon 2026 Submission

**Team:** APEXMIND
**Solution name:** VeriQuest AI
**Final answers:** [`APEXMIND_submission.csv`](APEXMIND_submission.csv)

## What this is

VeriQuest AI is an evidence-grounded pipeline that answers the 100 official
multiple-choice questions in `questions_100.csv`. Instead of asking an LLM to
guess from memory, it:

1. Builds several **targeted search queries** per question (not one giant
   query stuffed with every option).
2. Retrieves live web evidence from **DuckDuckGo**.
3. **Deduplicates** evidence by URL and by normalised title/snippet text.
4. **Ranks** evidence deterministically using keyword overlap, exact phrase
   matches, main-entity matches, and source authority (Wikipedia, Britannica,
   etc).
5. **Scores every option (A–E)** against the ranked evidence.
6. Builds a **RAG prompt** containing the question, options, ranked evidence,
   and the deterministic option scores.
7. Asks a **local <=8B LLM (via Ollama)** to act as an evidence judge and pick
   the best-supported option.
8. Combines the LLM's answer with the deterministic scorer using a
   **confidence score** — when the evidence overwhelmingly supports one option,
   the deterministic answer is trusted directly; otherwise the LLM's judgement
   is used; if neither is available, it falls back gracefully to `Unknown`.
9. **Validates** every answer before saving, and writes detailed **debug
   logs** (per-question evidence, scores, and reasoning) for inspection.

## Why evidence-grounded answering beats direct prompting

An 8B model asked to answer a Wikipedia trivia question from memory alone
will frequently hallucinate a plausible-sounding but wrong option, especially
for obscure topics (small towns, regional films, niche athletes). By first
retrieving real web evidence and forcing the model to **judge the evidence
rather than recall facts**, wrong answers are far more often caught: either
the deterministic scorer finds clear textual support for the correct option,
or the LLM sees the real Wikipedia snippet rather than relying on a fuzzy
memorized association. In manual spot-checks during development the pipeline
correctly answered every verifiable sample question (MARSOC capabilities,
George Anson's war, P. Padmarajan's filmmaking career, Edmond Malone's
Shakespeare scholarship, CoRoT's short-period exoplanet focus) — including
one case where the deterministic scorer hit a tie and correctly deferred to
the LLM judge reading the actual evidence.

## Model used (size compliance)

| | |
|---|---|
| Model | `llama3.1:8b` served locally via [Ollama](https://ollama.com) |
| Parameters | **8.0B** (confirmed via `ollama show llama3.1:8b`), quantized Q4_K_M |
| Rule | Hackathon requires <=8B parameters — this model is exactly at the limit and is the only model used to generate answers. |

The model is configurable via an environment variable so it can be swapped
for another <=8B model without touching code:

```bash
export HACKATHON_MODEL=qwen2.5:7b-instruct   # or any other <=8B Ollama model
```

If Ollama is not running or the model is unavailable, the pipeline prints a
clear warning and **automatically falls back to the deterministic
evidence-scoring answer** — it never crashes and always produces a valid CSV.

## Setup

This pipeline intentionally avoids `pandas`/`numpy` — they fail to build from
source on newer/pre-release Python versions (no upstream wheels yet), and
they aren't actually needed for 100 rows of CSV. All CSV I/O uses Python's
built-in `csv` module.

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r starter_code/requirements.txt
```

### Ollama setup (required for the LLM judge)

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.1:8b
# Ollama runs as a local server at http://localhost:11434 automatically
```

If you skip this step entirely, the pipeline still works — it just relies
solely on the deterministic evidence scorer (`--no-llm` makes this explicit).

## Test command (quick smoke test, 5 questions)

```bash
python starter_code/run.py --questions questions_100.csv --output TEAMNAME_submission.csv --limit 5
```

## Full run (all 100 questions, generates the official submission)

```bash
python starter_code/run.py --questions questions_100.csv --output APEXMIND_submission.csv
```

This takes roughly **15–30 minutes** depending on network speed and CPU,
because it performs 3–6 real web searches per question (up to ~500 search
calls total) plus one local LLM call per question. Progress is shown with a
live progress bar.

Useful optional flags:

```bash
--model qwen2.5:7b-instruct   # override the Ollama model for this run
--no-llm                      # disable the LLM judge, deterministic-only mode
```

## Output format

`APEXMIND_submission.csv` contains exactly:

```csv
question_no,answer
1,B
2,D
3,B
...
```

- Columns are exactly `question_no,answer`.
- Answers are strictly one of `A`, `B`, `C`, `D`, `E`, or `Unknown`.
- The pipeline validates this before writing the file and will raise an
  error rather than save an invalid submission.

## Debug logs

Every run also writes to `logs/` (auto-created, gitignored if you choose to
ignore it):

- `logs/predictions_debug.csv` — per-question final/LLM/deterministic
  answers, confidence, search queries used, option scores, and top evidence
  titles/URLs.
- `logs/evidence_log.jsonl` — one JSON object per question with the full
  ranked evidence, option scores, and decision reasoning, for deep debugging.

## Limitations

- DuckDuckGo search can rate-limit or return thin results for very obscure
  topics; when no evidence is found, the LLM is told to fall back to its own
  general knowledge but is encouraged to answer `Unknown` if unsure.
- The deterministic scorer uses keyword/phrase overlap rather than semantic
  embeddings — it is fast and explainable but can be fooled by options that
  share a lot of surface vocabulary. The confidence mechanism specifically
  detects tied/ambiguous deterministic scores and defers to the LLM in that
  case.
- Entity extraction (`extract_main_entity`) is a regex heuristic (capitalised
  word runs / quoted text), not a true NER model — it works well on
  Wikipedia-style "What/Who/Which..." questions but can occasionally pick the
  wrong noun phrase on more unusual phrasing.
- A full 100-question run depends on live internet search results, so
  re-running on a different day/network can yield slightly different
  evidence (and therefore occasionally different answers) than reproducible
  fully-offline pipelines would.

## Reproducibility notes

- The LLM is called with `temperature=0` for deterministic decoding given
  the same prompt/evidence.
- All non-LLM scoring (query generation, evidence ranking, option scoring,
  confidence) is fully deterministic pure-Python logic — re-running with the
  same evidence always produces the same scores.
- The only source of run-to-run variance is live web search results
  changing over time, which is inherent to "internet-assisted answering."

## Files

| File | Purpose |
|---|---|
| `starter_code/run.py` | The full VeriQuest AI pipeline (this is the upgraded starter code). |
| `starter_code/requirements.txt` | Minimal dependency list (`ddgs`, `requests`, `tqdm`). |
| `APEXMIND_submission.csv` | Final answers for all 100 questions. |
| `tests/test_pipeline.py` | Sanity checks for query generation, answer validation, and CSV output format. |
| `docs/PRESENTATION_NOTES.md` | Slide-ready notes on problem, architecture, and compliance. |
| `logs/` | Debug logs generated by each run. |
