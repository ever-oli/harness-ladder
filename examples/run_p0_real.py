from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness_ladder.config import ModelConfig
from harness_ladder.eval_runner import run_suite
from harness_ladder.ledger import append_row
from harness_ladder.model import TransformersLLM


def main() -> int:
    config = ModelConfig(model_id="openbmb/MiniCPM5-2B", seed=0, max_tokens=64)
    client = TransformersLLM(config, max_new_tokens=64)
    started = time.perf_counter()
    summary = run_suite(
        rung=0,
        suite_path=ROOT / "tasks" / "suite_smoke.json",
        client=client,
        config=config,
        write_ledger=False,
        notes="Lightning T4 fp16 real baseline",
    )
    wall = time.perf_counter() - started
    results = []
    for result in summary["results"]:
        trajectory = result.trajectory
        results.append({
            "task_id": result.task_id,
            "success": result.success,
            "expected": result.expected,
            "actual": result.actual,
            "failure_reason": "" if result.success else (result.notes or "checker mismatch"),
            "output_tokens": trajectory.tokens_used,
            "latency_s": trajectory.wall_time_s,
        })
    n = len(results)
    passed = sum(1 for item in results if item["success"])
    avg_latency = sum(item["latency_s"] for item in results) / n
    avg_tokens = sum(item["output_tokens"] for item in results) / n
    report = {
        "rung": "P0",
        "model_id": config.model_id,
        "seed": config.seed,
        "max_new_tokens": 64,
        "passed": passed,
        "n_tasks": n,
        "success_rate": passed / n if n else 0.0,
        "avg_latency_s": avg_latency,
        "avg_output_tokens": avg_tokens,
        "total_wall_s": wall,
        "tasks": results,
    }
    run_path = ROOT / "results" / "runs" / "p0_real_minicpm5.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    append_row({
        "rung": 0,
        "powers": "P0",
        "success_rate": report["success_rate"],
        "n_tasks": n,
        "model_id": config.model_id,
        "seed": config.seed,
        "notes": "Lightning T4 fp16 real baseline",
        "avg_latency_s": f"{avg_latency:.6f}",
        "avg_output_tokens": f"{avg_tokens:.3f}",
        "total_wall_s": f"{wall:.6f}",
    }, ROOT / "results" / "ledger.csv")
    print(json.dumps(report, indent=2))
    return 0 if passed == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
