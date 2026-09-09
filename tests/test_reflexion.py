from harness_ladder.config import PowerFlags
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM
from harness_ladder.refine import normalize_compact
from harness_ladder.reflexion import reflect, retry_with_reflection


def test_normalize_keeps_space_separated_range_numbers():
    assert (
        normalize_compact("1 2 3", prompt="range(1,4) prints which numbers? space-separated")
        == "1 2 3"
    )


def test_reflect_and_retry_with_mock():
    client = MockLLM()
    reflection = reflect(client, "60 km in 2 hours speed?", "The speed is 30 km/h.")
    assert "verbose" in reflection.lower() or "token" in reflection.lower() or reflection
    retried = retry_with_reflection(
        client, "60 km in 2 hours speed?", "The speed is 30 km/h.", reflection
    )
    assert retried == "30"


def test_p7_loop_emits_reflection_and_retry_markers():
    traj = run_v0_loop(
        "Reply with exactly: The speed is 30 km/h.",
        task_id="adhoc",
        client=MockLLM(),
        flags=PowerFlags.for_rung(7),
    )
    roles = "\n".join(m.content for m in traj.messages)
    assert "P7_REFLECTION:" in roles
    assert "P7_RETRY:" in roles
    assert "30" in traj.final_answer
