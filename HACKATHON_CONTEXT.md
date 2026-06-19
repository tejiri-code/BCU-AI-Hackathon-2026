# BCU AI Hackathon 2026 — context handoff (for the agent in this window)

> Read this first. Date: **2026-06-19**. Duration **4 hours** (started ~10:48, so finish ~**14:48**).
> Repo: https://github.com/tejiri-code/BCU-AI-Hackathon-2026 (cloned here).

## Challenge
Build a **Generative AI question-answering pipeline** that answers **100 multiple-choice questions**.
Allowed: prompting, internet-assisted search, embeddings, retrieval, ranking, RAG, any AI pipeline —
**provided the LLM is not larger than 8B parameters**. Goal: **maximise accuracy** with a **reproducible,
well-documented** solution.

## Rules
- Max model size: **8B parameters** (must be documented/justified).
- Team 1–3. Internet-assisted answering allowed.
- Submit **answer file + code + PowerPoint** to own GitHub repo; submit link via the Google/MS Form in Teams.

## Repo files
- `questions_100.csv` — the 100 MCQs (inspect columns first).
- `answer_template.csv` — fill this with predicted answers (match its exact format).
- `starter_code/`, `docs/`, `slides/`, `README.md`.

## User hardware + local models (already set up)
- **RTX 4060 Laptop, 8 GB VRAM**; Ryzen 7 7840HS. Runs **GGUF via prebuilt llama.cpp server** (OpenAI-compatible
  API). The thesis project at `C:\Users\nwagb\Desktop\thesis` has `tools/llamacpp/` (binaries) and `models/`.
- **≤8B models available locally** (in `...\thesis\models\`), all GGUF Q4:
  - `Qwen3-4B-Instruct-2507` (4B) — recommended default (fast, strong).
  - `Qwen3.5-4B` (4B), `gemma-4-E4B` (~4B), `ministral-3-3B` (3B).
  - `Qwen2.5-VL-7B-Instruct` (7B, vision), `Qwen3-VL-8B-Instruct` (8B, vision — at the cap; best accuracy if questions need more capacity or images).
- Over the 8B cap (DO NOT use): gemma-4-12b, Qwen2.5-Coder-14B, gemma-4-26B-A4B, Qwen3-30B-A3B, Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B.

## Recommended pipeline (fast path)
1. **Inspect** `questions_100.csv` + `answer_template.csv` (columns, answer format: letter vs text).
2. Start the **llama.cpp server** with a ≤8B model (e.g. Qwen3-4B-Instruct), temp **0** for determinism.
3. For each question: build a clear MCQ prompt (question + options), instruct it to **output only the letter**;
   parse robustly. Consider **self-consistency / majority vote** (a few samples) for hard ones.
4. **Internet-assisted (allowed):** add a retrieval step (web search or local embeddings over any provided
   context) for knowledge-heavy questions; feed top snippets into the prompt (lightweight RAG).
5. **Write** predictions into `answer_template.csv` exactly as required.
6. **Document** model choice + 8B compliance; make it reproducible (script + README + requirements).
7. Build the **PowerPoint** (approach, pipeline diagram, model + why ≤8B, accuracy, reproducibility).
8. Push to a **new GitHub repo**, submit the link via the form.

## Priorities under time pressure
Accuracy first via a solid prompt + determinism; add retrieval only if time allows. Keep the answer-file
format EXACT (auto-grading). Commit early and often.
