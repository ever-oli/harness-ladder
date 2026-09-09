"""Append/read results/ledger.csv — one row per suite run."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LEDGER_COLUMNS: Sequence[str] = (
    "rung",
    "powers",
    "success_rate",
    "n_tasks",
    "model_id",
    "seed",
    "notes",
    "avg_latency_s",
    "avg_output_tokens",
    "total_wall_s",
)

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parents[2] / "results" / "ledger.csv"


def ensure_ledger(path: Path | str | None = None) -> Path:
    path = Path(path) if path else DEFAULT_LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(LEDGER_COLUMNS))
            writer.writeheader()
    return path


def append_row(row: Mapping[str, Any], path: Path | str | None = None) -> Path:
    path = ensure_ledger(path)
    payload = {col: row.get(col, "") for col in LEDGER_COLUMNS}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LEDGER_COLUMNS))
        writer.writerow(payload)
    return path


def read_rows(path: Path | str | None = None) -> list[dict[str, str]]:
    path = ensure_ledger(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def format_row(
    *,
    rung: int,
    powers: str,
    success_rate: float,
    n_tasks: int,
    model_id: str,
    seed: int,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "rung": rung,
        "powers": powers,
        "success_rate": f"{success_rate:.4f}",
        "n_tasks": n_tasks,
        "model_id": model_id,
        "seed": seed,
        "notes": notes,
    }


def write_header_only(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LEDGER_COLUMNS))
        writer.writeheader()
    return path
