from harness_ladder.config import PowerFlags
from harness_ladder.long_horizon import suggest_expression
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM


def test_suggest_expression_start_double_add_sub():
    assert suggest_expression("start 5, double, add 3, subtract 4, final?") == "((5*2)+3)-4"


def test_suggest_expression_add_to_list_last():
    assert suggest_expression("add 10 to 1,2,3,4, last result?") == "4+10"


def test_long_horizon_loop_uses_forced_calculator():
    traj = run_v0_loop(
        "start 5, double, add 3, subtract 4, final?",
        task_id="v1_long_horizon_01",
        client=MockLLM(),
        flags=PowerFlags.for_rung(8),
        category="long_horizon",
    )
    assert traj.final_answer == "9"
    assert any("calculator required" in m.content or "full expression" in m.content for m in traj.messages)


def test_long_horizon_add_last():
    traj = run_v0_loop(
        "add 10 to 1,2,3,4, last result?",
        task_id="v1_long_horizon_03",
        client=MockLLM(),
        flags=PowerFlags.for_rung(8),
        category="long_horizon",
    )
    assert traj.final_answer == "14"
