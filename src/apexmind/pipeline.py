"""End-to-end pipeline: retrieve -> answer (Qwen3-8B) -> write submission CSV."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from . import data as data_mod
from . import solver
from .config import Config
from .llm_client import LlamaClient
from .retrieval import evidence as ev
from .server import llama_server


def run(
    cfg: Config,
    limit: Optional[int] = None,
    model_name: str = "primary",
    mode: str = "rag_confidence",
) -> Dict:
    """Run the pipeline over the first `limit` questions (or all) and write the CSV.

    `mode` controls answering (closed_book | rag | rag_confidence) and confidence scoring.
    The submission CSV format is unchanged regardless of mode.
    """
    questions = data_mod.load_questions(cfg.path("questions"))
    if limit is not None:
        questions = questions.head(limit)

    cache_dir = cfg.path("cache_dir")
    gen = cfg.generation
    rcfg = cfg.retrieval
    model = cfg.model(model_name)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.path("runs_dir") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: retrieval (network/CPU, no GPU) for every question, cached to disk.
    print(f"[1/2] Retrieving evidence for {len(questions)} question(s)...")
    evidences: Dict[int, Dict] = {}
    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="retrieve"):
        qno = int(row["question_no"])
        options = data_mod.options_of(row)
        evidences[qno] = ev.gather_evidence(qno, str(row["question"]), options, rcfg, cache_dir)

    # Stage 2: answer with the primary model (server loaded once).
    print(f"[2/2] Answering with {model['file']} (port {model['port']}), mode={mode}...")
    records: List[Dict] = []
    logs: List[Dict] = []
    conf_rows: List[Dict] = []
    llama = cfg.llama
    with llama_server(
        server_bin=llama["server_bin"],
        model_path=cfg.model_path(model_name),
        host=llama["host"],
        port=model["port"],
        n_gpu_layers=llama["n_gpu_layers"],
        ctx=model["ctx"],
        startup_timeout_s=llama["startup_timeout_s"],
        log_path=run_dir / "server.log",
    ):
        client = LlamaClient(host=llama["host"], port=model["port"])
        for _, row in tqdm(questions.iterrows(), total=len(questions), desc="answer"):
            qno = int(row["question_no"])
            options = data_mod.options_of(row)
            items = evidences[qno]["items"]
            res = solver.solve_question(
                client, str(row["question"]), options, items, gen,
                mode=mode, char_budget=rcfg["evidence_char_budget"],
            )
            records.append({"question_no": qno, "answer": res["answer"]})
            comp = res["components"]
            conf_rows.append({
                "question_no": qno,
                "answer": res["answer"],
                "confidence": res["confidence"],
                "parse_score": comp["parse_score"],
                "evidence_score": comp["evidence_score"],
                "support_score": comp["support_score"],
                "agreement": comp["agreement"],
                "n_evidence": comp["n_evidence"],
                "has_wikipedia": comp["has_wikipedia"],
                "rag_answer": res["rag_answer"],
                "cb_answer": res["cb_answer"],
            })
            logs.append({
                "question_no": qno,
                "answer": res["answer"],
                "parsed_from": res["parsed_from"],
                "confidence": res["confidence"],
                "evidence_quality": res["evidence_quality"],
                "top_evidence": res["top_evidence"],
                "raw": res["raw"],
            })

    out_path = cfg.path("output")
    df = data_mod.write_submission(records, out_path)

    # Confidence report (one row per question) for inspection / triage.
    pd.DataFrame(conf_rows).to_csv(run_dir / "confidence_report.csv", index=False)

    # Run manifest + per-question log for reproducibility / inspection.
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "mode": mode,
        "model": {"name": model_name, "file": model["file"], "params_b": model["params_b"]},
        "generation": gen,
        "retrieval": rcfg,
        "output": str(out_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "answers_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    val = data_mod.validate_submission(out_path, expected_rows=len(questions))
    return {"output": str(out_path), "run_dir": str(run_dir), "validation": val, "rows": len(df)}
