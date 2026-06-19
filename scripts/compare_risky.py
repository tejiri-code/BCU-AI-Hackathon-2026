"""Run selected questions through one or more configured models and save comparisons.

This is a lightweight reproducibility helper for the final verification pass. It does
not write the official submission file; it only writes a comparison CSV under
outputs/runs/.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from apexmind import data as data_mod
from apexmind.config import load_config
from apexmind.llm_client import LlamaClient
from apexmind.retrieval import evidence as ev
from apexmind.server import llama_server
from apexmind import solver


def parse_qnos(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare risky MCQs across models")
    parser.add_argument("--qnos", required=True, help="Comma-separated question numbers")
    parser.add_argument("--models", default="voter_a,voter_b", help="Comma-separated config model names")
    parser.add_argument("--config", default=None, help="Optional config YAML path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    qnos = parse_qnos(args.qnos)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    questions = data_mod.load_questions(cfg.path("questions"))
    questions = questions[questions["question_no"].isin(qnos)].copy()
    questions["question_no"] = questions["question_no"].astype(int)
    questions = questions.sort_values("question_no")

    cache_dir = cfg.path("cache_dir")
    rcfg = cfg.retrieval
    gen = cfg.generation
    llama = cfg.llama

    evidences = {}
    for _, row in questions.iterrows():
        qno = int(row["question_no"])
        evidences[qno] = ev.gather_evidence(
            qno, str(row["question"]), data_mod.options_of(row), rcfg, cache_dir
        )["items"]

    rows = []
    for model_name in model_names:
        model = cfg.model(model_name)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = cfg.path("runs_dir") / f"compare_{run_id}_{model_name}.log"
        with llama_server(
            server_bin=llama["server_bin"],
            model_path=cfg.model_path(model_name),
            host=llama["host"],
            port=model["port"],
            n_gpu_layers=llama["n_gpu_layers"],
            ctx=model["ctx"],
            startup_timeout_s=llama["startup_timeout_s"],
            log_path=log_path,
        ):
            client = LlamaClient(host=llama["host"], port=model["port"])
            for _, row in questions.iterrows():
                qno = int(row["question_no"])
                options = data_mod.options_of(row)
                res = solver.solve_question(
                    client,
                    str(row["question"]),
                    options,
                    evidences[qno],
                    gen,
                    mode="rag_confidence",
                    char_budget=rcfg["evidence_char_budget"],
                )
                rows.append({
                    "question_no": qno,
                    "model": model_name,
                    "answer": res["answer"],
                    "confidence": res["confidence"],
                    "rag_answer": res["rag_answer"],
                    "cb_answer": res["cb_answer"],
                    "agreement": res["components"]["agreement"],
                    "support_score": res["components"]["support_score"],
                    "n_evidence": res["components"]["n_evidence"],
                    "has_wikipedia": res["components"]["has_wikipedia"],
                })

    out_dir = cfg.path("runs_dir") / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "risky_model_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
