# ApexMind — Memory & Handoff

> Single-file handoff so any teammate (or a fresh Claude account) can continue this hackathon
> work with full context. Mirrors the agent's persistent memory. Date: **2026-06-19**.

---

## 1. User / environment
- **Hardware:** RTX 4060 Laptop GPU (**8 GB VRAM**), Ryzen 7 7840HS, Windows 11.
- **Inference engine:** prebuilt **llama.cpp server** (CUDA, OpenAI-compatible API) at
  `C:\Users\nwagb\Desktop\thesis\tools\llamacpp\bin\llama-server.exe`.
- **Local GGUF model store (source of truth):** `C:\Users\nwagb\Desktop\thesis\models\`.
- **Python preference:** use a **Python 3.11** venv (3.11.9). Project venv already created at `.venv`
  (`py -3.11 -m venv .venv`). Installed Pythons on machine: 3.11, 3.12, 3.13, 3.14.

## 2. The challenge (BCU AI Hackathon 2026)
- Build a **Generative-AI pipeline answering 100 MCQs** in `questions_100.csv` (5 options A–E),
  maximising accuracy. Questions are **Wikipedia-style knowledge** ("according to Wikipedia ...").
- Output: **`ApexMind_submission.csv`** with columns `question_no,answer`; values `A`–`E` or `Unknown`;
  **exactly 100 rows** (exact format — auto-graded).
- Hard rule: **LLM ≤ 8B parameters**, documented/justified.
- Internet-assisted answering (web search / RAG) is allowed.
- Deliverables: answer file + code + PowerPoint, pushed to a **new GitHub repo**; submit link via the
  Teams form (Dr Nouh Elmitwally — Nouh.Elmitwally@bcu.ac.uk).
- Original repo cloned from https://github.com/tejiri-code/BCU-AI-Hackathon-2026
- Fuller original handoff: see `HACKATHON_CONTEXT.md`.

## 3. Locked decisions
- **Team name:** `ApexMind` (→ `ApexMind_submission.csv`, README, slides).
- **Strategy:** max-accuracy ensemble (Wikipedia+web RAG → rerank → Qwen3-8B primary + self-consistency →
  multi-model vote on low-confidence questions).
- **Evaluation:** inter-model/method agreement + confidence on all 100, **plus** a ~20-question manual
  gold set (`data/gold_dev.csv`) hand-verified against Wikipedia for an accuracy estimate + ablations.
- **GitHub:** build the repo locally, **no remote push until the user explicitly approves**.

## 4. Models available locally (all in this repo's `models/`, all ≤8B)
| Role | File | Params |
|---|---|---|
| **Primary reasoner** | `Qwen3-8B-Q4_K_M.gguf` | 8B (at cap) |
| Ensemble voter | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 4B |
| Ensemble voter | `Qwen3.5-4B-Q4_K_M.gguf` | 4B |
| Embeddings (RAG) | `Qwen3-Embedding-0.6B-f16.gguf` | 0.6B |
| Reranker (cross-encoder) | `bge-reranker-v2-m3-Q8_0.gguf` | 0.5B |
| Fallback | `gemma-4-E4B-it-Q4_K_M.gguf` | ~4B |
| Fallback | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` (+mmproj) | 3B |
| Vision (unused) | `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` (+mmproj) | 8B |
| Vision (unused) | `Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (+mmproj) | 7B |
| Embeddings (alt) | `embeddinggemma-300M-BF16.gguf` | 0.3B |

**Over the 8B cap — DO NOT USE:** gemma-4-12b, gemma-4-26B-A4B, Qwen2.5-Coder-14B, Qwen3-30B-A3B,
Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B, Qwen3-Coder-Next (these stay in the thesis folder, not copied here).

Re-download the extra models with `python scripts/fetch_models.py` (Qwen3-8B, Qwen3-Embedding-0.6B,
bge-reranker-v2-m3). Local-only models were copied from the thesis store (section 1).

## 5. Status (as of handoff)
- ✅ Python 3.11 venv at `.venv`; all 13 model files in `models/` (~30 GB).
- ✅ Plan approved and saved at `docs/PLAN.md` (full architecture + build order + verification).
- ⏸️ **Implementation not started yet** — awaiting user "go". Build order is in `docs/PLAN.md`.

## 6. How to resume (for a teammate / new account)
1. Read `docs/PLAN.md` (architecture + step-by-step build order) and this file.
2. Activate venv: `.\.venv\Scripts\Activate.ps1` (Windows PowerShell).
3. Follow the "Build order" in `docs/PLAN.md` (scaffold → deps → model/server/client → retrieval →
   solver → ensemble → pipeline → run 100 → evaluate → submission → README/slides → tests).
4. Keep the LLM ≤ 8B and the submission format exact. Do not push to GitHub without the user's OK.
