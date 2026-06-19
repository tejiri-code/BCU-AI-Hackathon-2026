# VeriQuest AI — Presentation Notes

Team: **APEXMIND**

## Starter code TODO completion (slide-ready checklist)

The original `starter_code/run.py` had four explicit `# TODO` sections and an
8-step "suggested improvements" list at the bottom. All of them are done:

| Starter code TODO | Status | Where it lives now |
|---|---|---|
| `build_search_query()` — "Improve this function... generate multiple queries" | **Done** | `build_search_queries()` (run.py:239) — 3-6 targeted queries instead of one mega-query: cleaned question, quoted entity, entity+property, entity+wikipedia, 1-2 distinctive-option-keyword queries |
| `rank_evidence()` — "Implement your ranking method... keyword overlap / TF-IDF / embeddings / cross-encoder" | **Done** | `rank_evidence()` + `score_evidence_item()` (run.py:367, 416) — deterministic keyword overlap, exact-phrase, entity-match, title-relevance, and source-authority scoring (chose explainable keyword scoring over embeddings — see "what we'd improve") |
| `build_prompt()` — "Improve this prompt" | **Done** | `build_prompt()` (run.py:586) — full RAG prompt: question, options, ranked evidence with title/snippet/URL, deterministic option scores, and a strict single-token output instruction |
| `generate_answer()` — "Replace this placeholder [`return "A"`] with your own model" | **Done** | `call_ollama()` (run.py:639) — real local `llama3.1:8b` (8.0B) call via Ollama's HTTP API, `temperature=0` |
| *(implicit)* answer validation | **Done & hardened** | `clean_answer()` (run.py:681) — handles bare letters, "The answer is B.", "Unknown", never returns blank |
| *(implicit)* confidence scoring | **Done** | `compute_confidence()` (run.py:548) — combines top score, gap-to-runner-up, evidence quantity, exact-match flag; arbitrates LLM vs deterministic answer |
| *(implicit)* evidence/debug logging | **Done** | `write_debug_logs()` (run.py:804) → `logs/predictions_debug.csv` + `logs/evidence_log.jsonl`; `review_report.py` surfaces the riskiest answers for manual spot-checking |
| Suggested step 1: "Test DuckDuckGo retrieval on 3-5 questions" | **Done** | `--limit 5` smoke test, run and verified |
| Suggested step 7-8: "Test on questions" / "Run on the full question set" | **Done** | Full 100-question run produces `APEXMIND_submission.csv`, validated (exactly `question_no,answer`, all answers in `{A,B,C,D,E,Unknown}`, 100 rows) |

## Problem

Answer 100 multiple-choice trivia/Wikipedia-style questions using a
Generative AI pipeline, with a hard constraint: the answering LLM must be
**<=8B parameters**. Direct prompting of a small model from memory alone is
unreliable on obscure topics (small towns, regional films, niche people) —
small models hallucinate confidently.

## Solution

VeriQuest AI: a retrieval-augmented pipeline where the LLM acts as an
**evidence judge**, not a memory-based guesser.

```
question
  -> build_search_queries()        3-6 targeted queries (not one giant query)
  -> DuckDuckGo search              ~5 results per query
  -> dedupe (URL + text fingerprint)
  -> rank_evidence()                deterministic keyword/phrase/source scoring
  -> score_options()                A-E support scores from ranked evidence
  -> build_prompt()                 RAG prompt: question + options + evidence + scores
  -> Ollama (<=8B local LLM)        evidence-grounded judgement
  -> decide_final_answer()          confidence-weighted: deterministic vs LLM
  -> clean_answer()                 strict validation -> A/B/C/D/E/Unknown
  -> CSV export + debug logs
```

## Architecture decisions

- **Multiple short queries, not one mega-query.** The starter code joined the
  question and all five options into a single search string, which dilutes
  relevance. We generate a cleaned question query, a quoted main-entity
  query, an entity+property query, an entity+"wikipedia" query, and 1-2
  distinctive-option-keyword queries.
- **Deterministic ranking before the LLM ever sees evidence.** Keyword
  overlap, exact phrase matches, main-entity presence, title relevance, and a
  source-authority bonus (Wikipedia/Britannica/etc) combine into a
  transparent relevance score — fully explainable, no embeddings/black box
  needed for this scale.
- **Option scoring is independent of the LLM.** Each option gets a support
  score purely from textual overlap with the top evidence. This gives a
  second, fully deterministic "opinion" that can be compared against the
  LLM's answer.
- **Confidence arbitrates between the two opinions.** Confidence combines the
  top option's score, the gap to the runner-up, evidence quantity, and
  whether an exact phrase match was found — critically, a **tie** between two
  options is explicitly penalised (a tie means the deterministic scorer
  cannot discriminate, so the LLM's reading of the actual evidence is trusted
  instead). High confidence overrides the LLM; otherwise the LLM's judgement
  is used; if both are absent, falls back to `Unknown`.
- **Everything degrades gracefully.** No search results -> empty evidence
  list, not a crash. No Ollama -> deterministic-only mode with a clear
  warning. Malformed LLM output -> `clean_answer()` extracts a letter or
  defaults to `Unknown`, never blank.

## Model rule compliance

- Model: **`llama3.1:8b`**, served locally via Ollama.
- Size: **8.0B parameters** (verified with `ollama show llama3.1:8b`), Q4_K_M
  quantisation.
- This is the *only* model used to produce answers in the submitted
  pipeline — no larger model (including Claude, used only as the coding
  assistant during development) ever generates an answer.
- Swappable via `HACKATHON_MODEL` env var or `--model` CLI flag to any other
  <=8B Ollama model (e.g. `qwen2.5:7b-instruct`).

## Why evidence-grounded answering reduces hallucination

A small instruction-tuned model has limited and sometimes garbled long-tail
factual recall. Two things specifically reduce hallucination here:

1. **Grounding in retrieved text.** The model is shown actual Wikipedia/news
   snippets relevant to the question instead of being asked to recall facts
   unaided — it only has to *read and match*, a much easier task for an 8B
   model than free recall.
2. **A deterministic second opinion.** Even when the LLM does hallucinate,
   the keyword/phrase-overlap option scorer independently looks at the same
   evidence and frequently catches the correct option from raw textual
   support, and confidence-weighting lets that catch override a wrong LLM
   guess when the evidence signal is strong and unambiguous.

## Debugging story (good evidence of rigor for Q&A)

Two real scoring bugs were found and fixed during testing by manually
checking individual answers against their evidence logs — useful to mention
if asked "how do you know your answers are good":

1. **Q4** ("length of U.S. Route 11 in Georgia"): the route number "11" is
   repeated in nearly every search result *and* happens to be the leading
   digit of option A ("11 miles"). Plain keyword overlap let that coincidence
   outscore option D, even though the top Wikipedia result said "22.8-mile-
   long (36.7 km)" verbatim — exactly option D. **Fix:** exclude digit tokens
   shared between the question and evidence from option scoring, plus add a
   dedicated numeric exact-match check (`extract_numbers()`, run.py:446).
2. **Q25** ("purpose of the Liberty L-6 engine"): the first fix above was
   *too* aggressive — it also stripped the legitimate word "aircraft" from
   option A because the question itself contains "aircraft" (in "...Liberty
   L-6 **aircraft** engine..."). Meanwhile a "distinctive token" bonus
   over-rewarded option D ("fighter jets") for matching the word "fighter" in
   an unrelated evidence phrase ("fighter aircraft", a WWI biplane, not a
   jet). **Fix:** narrowed the exclusion to digits only (not all shared
   words), and reduced the distinctive-token bonus weight so it can't
   override a clearly better overlap-fraction signal.

In both cases **the LLM judge had already read the evidence correctly** —
the bug was in the deterministic scorer overriding it via the
high-confidence shortcut. This is good evidence that the RAG+LLM-judge
design is sound; the deterministic layer needed tuning, not the model.

## How we verify answers without the answer key

The hackathon rules forbid using/possessing the organiser answer key, so
there's no compliant way to auto-grade. Instead: `starter_code/review_report.py`
ranks all 100 answers by risk (low confidence, or disagreement between the
LLM and the deterministic scorer) and prints the top evidence URLs for each,
so the riskiest ~15-20 can be manually checked against the real Wikipedia
pages instead of guessing which ones might be wrong.

## What we would improve with more time

- Swap keyword-overlap ranking for a small sentence-embedding reranker
  (still <=8B-compatible, e.g. a local cross-encoder) for semantic matches
  that don't share exact vocabulary.
- Add a second-pass "self-check" prompt where the LLM is shown its own answer
  plus the runner-up option and asked to confirm or revise.
- Cache search results across runs to make iteration during development
  faster and reduce DuckDuckGo rate-limit risk on repeated full runs.
- Add per-question source-credibility weighting tuned against a held-out
  validation set (we don't have ground truth, so this was done by manual
  spot-checking instead).
