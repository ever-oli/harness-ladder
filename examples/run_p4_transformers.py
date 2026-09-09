#!/usr/bin/env python3
"""Run cumulative P4/P5 (or any rung) against local Transformers MiniCPM5-2B."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from harness_ladder.config import DEFAULT_MODEL_ID, ModelConfig
from harness_ladder.eval_runner import run_suite
from harness_ladder.model import TransformersLLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", type=int, default=5)
    parser.add_argument("--suite", type=Path, default=ROOT / "tasks" / "suite_v1_mini.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results" / "ledger.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "runs" / "p5_real_minicpm5_mini_v1.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    config = ModelConfig(model_id=args.model_id, seed=args.seed, max_tokens=args.max_new_tokens)
    client = TransformersLLM(config, max_new_tokens=args.max_new_tokens)
    summary = run_suite(
        rung=args.rung,
        suite_path=args.suite,
        client=client,
        config=config,
        ledger_path=args.ledger,
        write_ledger=True,
        notes=f"Lightning T4 fp16 cumulative P{args.rung} evaluation",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rung": summary["rung"],
        "powers": summary["powers"],
        "model_id": summary["model_id"],
        "seed": summary["seed"],
        "suite": str(args.suite),
        "n_tasks": summary["n_tasks"],
        "n_success": summary["n_success"],
        "success_rate": summary["success_rate"],
        "avg_tokens": summary["avg_tokens"],
        "avg_wall_time_s": summary["avg_wall_time_s"],
        "total_wall_time_s": summary["total_wall_time_s"],
        "results": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "expected": r.expected,
                "actual": r.actual,
                "tokens_used": r.trajectory.tokens_used,
                "wall_time_s": r.trajectory.wall_time_s,
                "messages": [asdict(m) for m in r.trajectory.messages],
            }
            for r in summary["results"]
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"rung={summary['rung']} powers={summary['powers']} "
        f"passed={summary['n_success']}/{summary['n_tasks']} "
        f"success_rate={summary['success_rate']:.2%}"
    )
    for r in summary["results"]:
        mark = "PASS" if r.success else "FAIL"
        print(f"  [{mark}] {r.task_id}: got={r.actual!r} expected={r.expected!r}")
    print(f"saved={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
