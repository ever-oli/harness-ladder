"""Tests for eval runner + suite smoke path."""

from pathlib import Path

from harness_ladder.config import ModelConfig
from harness_ladder.eval_runner import grade, load_suite, run_suite
from harness_ladder.ledger import read_rows
from harness_ladder.model import MockLLM


def test_grade_normalizes_whitespace():
    assert grade("OK", " OK ")
    assert not grade("OK", "NO")


def test_load_suite_smoke():
    suite = load_suite()
    assert 5 <= len(suite) <= 10
    assert all({"id", "prompt", "expected"} <= set(t) for t in suite)


def test_run_suite_mock_writes_ledger(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    config = ModelConfig(model_id="openbmb/MiniCPM5-2B", seed=7)
    summary = run_suite(
        rung=0,
        client=MockLLM(config),
        config=config,
        ledger_path=ledger,
        write_ledger=True,
        notes="pytest",
    )
    assert summary["n_tasks"] >= 5
    assert summary["success_rate"] == 1.0
    assert summary["powers"] == "P0"
    assert summary["model_id"] == "openbmb/MiniCPM5-2B"

    rows = read_rows(ledger)
    assert len(rows) == 1
    assert rows[0]["rung"] == "0"
    assert rows[0]["notes"] == "pytest"
    assert float(rows[0]["success_rate"]) == 1.0


def test_run_suite_without_ledger():
    summary = run_suite(rung=0, write_ledger=False, notes="no-ledger")
    assert summary["success_rate"] == 1.0
    assert all(r.success for r in summary["results"])
