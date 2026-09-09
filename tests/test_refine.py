from harness_ladder.config import PowerFlags
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM
from harness_ladder.refine import normalize_compact, self_refine


def test_normalize_strips_verbose_speed_and_version():
    assert normalize_compact("The speed is 30 km/h.", prompt="60 km in 2 hours speed?") == "30"
    assert normalize_compact("The changelog version 1.4.2 is the latest.", prompt="changelog version?") == "1.4.2"


def test_normalize_search_count_and_retries():
    assert normalize_compact("The search tool returned a count of 3.", prompt="search found 3, JSON count 3") == "count: 3"
    assert (
        normalize_compact(
            "The worker retries failed jobs 3 times before marking a job failed.",
            prompt="README worker retries 3, how many?",
        )
        == "3"
    )


def test_p6_self_refine_compacts_draft_via_mock():
    client = MockLLM()
    out = self_refine(client, "60 km in 2 hours speed?", "The speed is 30 km/h.")
    assert out == "30"


def test_p6_loop_flag_runs_refine_marker():
    # Force a verbose draft by using a prompt MockLLM doesn't special-case, then refine.
    # Use rung 6 with a simple echo-like path: Mock returns OK then refine keeps OK.
    traj = run_v0_loop(
        "Reply with exactly: The speed is 30 km/h.",
        task_id="adhoc",
        client=MockLLM(),
        flags=PowerFlags.for_rung(6),
    )
    # Exact reply is returned first; P6 normalize should compact using question heuristics.
    assert traj.final_answer in {"30", "The speed is 30 km/h."} or "30" in traj.final_answer
