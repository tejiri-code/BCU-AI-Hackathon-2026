"""Gold-set evaluation: accuracy, error inspection, confidence, and a simple ablation."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

import pandas as pd

from . import data as data_mod
from . import solver
from .config import Config
from .llm_client import LlamaClient
from .retrieval import evidence as ev
from .server import llama_server


def _solve_all(cfg: Config, gold: pd.DataFrame, model_name: str = "primary") -> List[Dict]:
    """Retrieve evidence and run a full rag_confidence solve for each gold question.

    rag_confidence yields both the RAG and closed-book answers, so every ablation
    mode can be derived from a single pass (no extra model calls)."""
    cache_dir = cfg.path("cache_dir")
    gen = cfg.generation
    rcfg = cfg.retrieval
    model = cfg.model(model_name)
    llama = cfg.llama

    # Stage 1: retrieval (cached).
    print(f"[1/2] Retrieving evidence for {len(gold)} gold question(s)...")
    items_by_q: Dict[int, List[Dict]] = {}
    for _, row in gold.iterrows():
        qno = int(row["question_no"])
        options = data_mod.options_of(row)
        items_by_q[qno] = ev.gather_evidence(
            qno, str(row["question"]), options, rcfg, cache_dir
        )["items"]

    # Stage 2: solve (server loaded once).
    print(f"[2/2] Solving with {model['file']} (port {model['port']})...")
    results: List[Dict] = []
    with llama_server(
        server_bin=llama["server_bin"],
        model_path=cfg.model_path(model_name),
        host=llama["host"],
        port=model["port"],
        n_gpu_layers=llama["n_gpu_layers"],
        ctx=model["ctx"],
        startup_timeout_s=llama["startup_timeout_s"],
    ):
        client = LlamaClient(host=llama["host"], port=model["port"])
        for _, row in gold.iterrows():
            qno = int(row["question_no"])
            options = data_mod.options_of(row)
            res = solver.solve_question(
                client, str(row["question"]), options, items_by_q[qno], gen,
                mode="rag_confidence", char_budget=rcfg["evidence_char_budget"],
            )
            res.update({
                "question_no": qno,
                "question": str(row["question"]),
                "gold": str(row["gold_answer"]).strip().upper(),
            })
            results.append(res)
    return results


def _accuracy(preds: List[str], golds: List[str]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return round(100.0 * correct / len(preds), 1)


def run(cfg: Config, ablation: bool = False, model_name: str = "primary") -> Dict:
    gold_path = cfg.path("gold")
    if data_mod.ensure_gold_template(
        cfg.path("questions"), gold_path, cfg.evaluation.get("gold_template_size", 20)
    ):
        print(f"Created gold template at {gold_path}.")
        print("Fill the 'gold_answer' column (A-E) for the questions you verify, then re-run.")
        return {"created_template": True, "labeled": 0}

    gold = data_mod.load_gold(gold_path)
    if gold.empty:
        print(f"No labeled rows in {gold_path} (gold_answer must be A-E). Fill some and re-run.")
        return {"created_template": False, "labeled": 0}

    results = _solve_all(cfg, gold, model_name=model_name)
    golds = [r["gold"] for r in results]

    rag_preds = [r["rag_answer"] for r in results]
    cb_preds = [r["cb_answer"] for r in results]
    thr = cfg.evaluation.get("confidence_threshold", 0.6)

    main_acc = _accuracy(rag_preds, golds)
    print("\n" + "=" * 60)
    print(f"GOLD EVALUATION  ({len(results)} labeled questions, mode=rag_confidence)")
    print("=" * 60)
    print(f"Accuracy: {main_acc}%  ({sum(p==g for p,g in zip(rag_preds,golds))}/{len(results)})")

    wrong = [r for r in results if r["rag_answer"] != r["gold"]]
    if wrong:
        print(f"\nWrong answers ({len(wrong)}):")
        for r in wrong:
            te = r["top_evidence"][0]["snippet"][:160] if r["top_evidence"] else ""
            print(f"  Q{r['question_no']}: predicted={r['rag_answer']} gold={r['gold']} "
                  f"confidence={r['confidence']}")
            print(f"      top evidence: {te}")
    else:
        print("\nAll labeled questions correct.")

    report = {
        "created_template": False,
        "labeled": len(results),
        "accuracy_rag": main_acc,
        "wrong": [
            {"question_no": r["question_no"], "predicted": r["rag_answer"],
             "gold": r["gold"], "confidence": r["confidence"]}
            for r in wrong
        ],
    }

    if ablation:
        hi = [r for r in results if r["confidence"] >= thr]
        hi_acc = _accuracy([r["rag_answer"] for r in hi], [r["gold"] for r in hi])
        print("\n" + "-" * 60)
        print("ABLATION (same questions; answers derived from one pass)")
        print("-" * 60)
        print(f"  {'mode':<18}{'n':>4}{'accuracy':>12}")
        print(f"  {'closed_book':<18}{len(results):>4}{_accuracy(cb_preds, golds):>11}%")
        print(f"  {'rag':<18}{len(results):>4}{main_acc:>11}%")
        print(f"  {'rag+confidence':<18}{len(results):>4}{main_acc:>11}%   "
              f"(high-conf >= {thr}: n={len(hi)}, acc={hi_acc}%)")
        report["ablation"] = {
            "closed_book": _accuracy(cb_preds, golds),
            "rag": main_acc,
            "rag_confidence_highconf": {"n": len(hi), "accuracy": hi_acc, "threshold": thr},
        }

    # Persist a detailed gold-eval report next to the other run artifacts.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.path("runs_dir") / f"eval_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "question_no": r["question_no"], "gold": r["gold"],
            "rag_answer": r["rag_answer"], "cb_answer": r["cb_answer"],
            "correct": r["rag_answer"] == r["gold"], "confidence": r["confidence"],
            "agreement": r["components"]["agreement"],
            "support_score": r["components"]["support_score"],
            "n_evidence": r["components"]["n_evidence"],
        }
        for r in results
    ]).to_csv(run_dir / "gold_eval.csv", index=False)
    (run_dir / "gold_eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nReport written to {run_dir}")
    return report
