#!/usr/bin/env python3
"""P0 smoke: run suite_smoke.json with MockLLM and print metrics."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install when PYTHONPATH includes src
ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from harness_ladder.config import ModelConfig
from harness_ladder.eval_runner import run_suite
from harness_ladder.model import MockLLM


def main() -> int:
    config = ModelConfig(model_id="openbmb/MiniCPM5-2B", seed=0)
    summary = run_suite(
        rung=0,
        client=MockLLM(config),
        config=config,
        write_ledger=True,
        notes="example p0 smoke",
    )
    print(
        f"rung={summary['rung']} powers={summary['powers']} "
        f"success_rate={summary['success_rate']:.2%} "
        f"n_tasks={summary['n_tasks']} model={summary['model_id']}"
    )
    for r in summary["results"]:
        mark = "PASS" if r.success else "FAIL"
        print(f"  [{mark}] {r.task_id}: got={r.actual!r} expected={r.expected!r}")
    return 0 if summary["success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
