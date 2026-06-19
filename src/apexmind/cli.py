"""Command-line entry point for the ApexMind pipeline."""
from __future__ import annotations

import argparse

from .config import load_config
from . import pipeline, evaluate
from .solver import MODES


def main() -> None:
    parser = argparse.ArgumentParser(prog="apexmind", description="ApexMind MCQ QA pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Retrieve + answer questions and write the submission CSV")
    run_p.add_argument("--limit", type=int, default=None, help="Answer only the first N questions")
    run_p.add_argument("--config", default=None, help="Path to a YAML config (default: config/default.yaml)")
    run_p.add_argument("--model", default="primary", help="Model registry name (default: primary)")
    run_p.add_argument("--mode", default="rag_confidence", choices=MODES,
                       help="Answering mode (default: rag_confidence)")

    eval_p = sub.add_parser("evaluate", help="Run accuracy evaluation on data/gold_dev.csv")
    eval_p.add_argument("--config", default=None, help="Path to a YAML config")
    eval_p.add_argument("--model", default="primary", help="Model registry name (default: primary)")
    eval_p.add_argument("--ablation", action="store_true",
                        help="Also print a closed_book vs rag vs rag+confidence comparison")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "evaluate":
        evaluate.run(cfg, ablation=args.ablation, model_name=args.model)
        return

    if args.command == "run":
        result = pipeline.run(cfg, limit=args.limit, model_name=args.model, mode=args.mode)
        print("\n=== Result ===")
        print(f"Output : {result['output']}  ({result['rows']} rows)")
        print(f"Run dir: {result['run_dir']}")
        val = result["validation"]
        status = "OK" if val["ok"] else "PROBLEMS"
        print(f"Validation: {status}")
        for p in val["problems"]:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
