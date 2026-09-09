"""Tests for ledger append/read."""

from pathlib import Path

from harness_ladder.ledger import (
    LEDGER_COLUMNS,
    append_row,
    ensure_ledger,
    format_row,
    read_rows,
    write_header_only,
)


def test_ensure_and_append(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    ensure_ledger(path)
    rows = read_rows(path)
    assert rows == []

    row = format_row(
        rung=0,
        powers="P0",
        success_rate=1.0,
        n_tasks=8,
        model_id="openbmb/MiniCPM5-2B",
        seed=0,
        notes="unit",
    )
    append_row(row, path=path)
    rows = read_rows(path)
    assert len(rows) == 1
    assert rows[0]["rung"] == "0"
    assert rows[0]["powers"] == "P0"
    assert rows[0]["success_rate"] == "1.0000"
    assert rows[0]["n_tasks"] == "8"
    assert rows[0]["model_id"] == "openbmb/MiniCPM5-2B"
    assert list(rows[0].keys()) == list(LEDGER_COLUMNS)


def test_write_header_only(tmp_path: Path):
    path = tmp_path / "empty.csv"
    write_header_only(path)
    text = path.read_text(encoding="utf-8").strip()
    assert text == ",".join(LEDGER_COLUMNS)
