"""Run a fixed task suite with selected power flags and append a ledger row."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from harness_ladder.config import ModelConfig, PowerFlags
from harness_ladder.ledger import append_row, format_row
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import LLMClient, MockLLM
from harness_ladder.types import TaskResult

DEFAULT_SUITE = Path(__file__).resolve().parents[2] / "tasks" / "suite_smoke.json"


def load_suite(path: Path | str | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else DEFAULT_SUITE
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Suite must be a non-empty list: {path}")
    return data


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def grade(expected: str, actual: str, match: str = "exact") -> bool:
    if match == "exact": return _normalize(expected) == _normalize(actual)
    if match == "contains": return _normalize(expected) in _normalize(actual)
    if match == "regex": return re.search(expected, actual, re.I|re.S) is not None
    raise ValueError(f"Unsupported match mode: {match}")


def run_task(
    task: dict[str, Any],
    *,
    client: LLMClient,
    config: ModelConfig,
    flags: PowerFlags,
) -> TaskResult:
    traj = run_v0_loop(
        task["prompt"],
        task_id=task["id"],
        client=client,
        config=config,
        flags=flags,
        category=task.get("category"),
        tags=task.get("tags"),
    )
    ok = grade(task["expected"], traj.final_answer, task.get("match", "exact"))
    return TaskResult(
        task_id=task["id"],
        success=ok,
        expected=task["expected"],
        actual=traj.final_answer,
        trajectory=traj,
    )


def run_suite(
    *,
    rung: int = 0,
    suite_path: Path | str | None = None,
    client: Optional[LLMClient] = None,
    config: Optional[ModelConfig] = None,
    ledger_path: Path | str | None = None,
    write_ledger: bool = True,
    notes: str = "",
    tasks: Optional[Sequence[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run suite with cumulative flags for ``rung``; optionally append ledger."""
    config = config or ModelConfig()
    flags = PowerFlags.for_rung(rung)
    client = client or MockLLM(config)
    suite = list(tasks) if tasks is not None else load_suite(suite_path)

    results = [
        run_task(task, client=client, config=config, flags=flags) for task in suite
    ]
    n = len(results)
    n_ok = sum(1 for r in results if r.success)
    success_rate = (n_ok / n) if n else 0.0

    summary: dict[str, Any] = {
        "rung": rung,
        "powers": flags.as_csv(),
        "success_rate": success_rate,
        "n_tasks": n,
        "n_success": n_ok,
        "model_id": config.model_id,
        "seed": config.seed,
        "results": results,
        "avg_tokens": (sum(r.trajectory.tokens_used for r in results) / n) if n else 0.0,
        "avg_wall_time_s": (sum(r.trajectory.wall_time_s for r in results) / n) if n else 0.0,
        "total_wall_time_s": sum(r.trajectory.wall_time_s for r in results),
    }

    if write_ledger:
        row = format_row(
            rung=rung,
            powers=flags.as_csv(),
            success_rate=success_rate,
            n_tasks=n,
            model_id=config.model_id,
            seed=config.seed,
            notes=notes or f"smoke rung={rung}",
        )
        append_row(row, path=ledger_path)

    return summary
