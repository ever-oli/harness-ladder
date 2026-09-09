#!/usr/bin/env python3
"""Run the real cumulative P1–P6 suite against MiniCPM5-2B.

The client is deliberately OpenAI-compatible (no MockLLM): start the live
Lightning/vLLM endpoint first, then run this script from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from harness_ladder.config import DEFAULT_MODEL_ID, ModelConfig
from harness_ladder.eval_runner import run_suite
from harness_ladder.model import OpenAICompatibleClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", type=int, default=1)
    parser.add_argument("--suite", type=Path, default=ROOT / "tasks" / "suite_smoke.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results" / "ledger.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "runs" / "p4_real_minicpm5_mini_v1.json")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rung not in (1, 2, 3, 4, 5, 6):
        raise SystemExit("This runner supports cumulative rungs 1-6 (P1-P6).")
    config = ModelConfig(model_id=args.model_id, base_url=args.base_url, seed=args.seed, max_tokens=128)
    summary = run_suite(
        rung=args.rung,
        suite_path=args.suite,
        client=OpenAICompatibleClient(config),
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
        "base_url": args.base_url,
        "n_tasks": summary["n_tasks"],
        "n_success": summary["n_success"],
        "success_rate": summary["success_rate"],
        "avg_tokens": summary["avg_tokens"],
        "avg_latency_s": summary["avg_wall_time_s"],
        "avg_wall_time_s": summary["avg_wall_time_s"],
        "total_wall_time_s": summary["total_wall_time_s"],
        "results": [
            {
                "task_id": result.task_id,
                "success": result.success,
                "expected": result.expected,
                "actual": result.actual,
                "tokens_used": result.trajectory.tokens_used,
                "wall_time_s": result.trajectory.wall_time_s,
                "messages": [asdict(message) for message in result.trajectory.messages],
            }
            for result in summary["results"]
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"rung={summary['rung']} powers={summary['powers']} passed={summary['n_success']}/{summary['n_tasks']} "
        f"success_rate={summary['success_rate']:.2%} "
        f"avg_tokens={summary['avg_tokens']:.2f} "
        f"avg_latency_s={summary['avg_wall_time_s']:.6f} "
        f"total_wall_time_s={summary['total_wall_time_s']:.6f}"
    )
    for result in summary["results"]:
        mark = "PASS" if result.success else "FAIL"
        print(f"  [{mark}] {result.task_id}: got={result.actual!r} expected={result.expected!r}")
    print(f"saved={args.output}")
    return 0 if summary["success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
