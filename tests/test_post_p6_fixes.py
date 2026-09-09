from harness_ladder.config import PowerFlags
from harness_ladder.eval_runner import grade
from harness_ladder.loop import run_v0_loop, _prefer_retry
from harness_ladder.model import MockLLM
from harness_ladder.refine import normalize_compact
from harness_ladder.repl import PersistentPythonREPL


def test_repl_print_does_not_append_none():
    assert PersistentPythonREPL().run("print(7 + 5)") == "12"


def test_normalize_strips_currency_and_units():
    assert normalize_compact("$33", prompt="$40 minus $7?") == "33"
    assert normalize_compact("4 apples", prompt="24 apples / 6 children?") == "4"


def test_prefer_retry_rejects_unit_expansion_and_random_rewrites():
    assert _prefer_retry("$40 minus $7?", "33", "$33") == "33"
    assert _prefer_retry("24 apples / 6 children?", "4", "4 apples") == "4"
    assert _prefer_retry("add(7,5)", "12", "7") == "12"  # equal length rewrite → keep draft
    assert _prefer_retry("x", "The speed is 30 km/h.", "30") == "30"  # strictly shorter OK
    assert _prefer_retry("x", "12", "Final 12 units") == "12"


def test_tool_05_regex_accepts_count_colon():
    assert grade(r"count\s*[:=]\s*3", "count: 3", "regex")


def test_p7_keeps_compact_answer_against_bad_retry_pressure():
    # MockLLM exact reply then P7 may try to expand; prefer_retry + normalize should keep 33-ish
    traj = run_v0_loop(
        "Reply with exactly: 33",
        task_id="adhoc",
        client=MockLLM(),
        flags=PowerFlags.for_rung(7),
    )
    assert traj.final_answer == "33"
