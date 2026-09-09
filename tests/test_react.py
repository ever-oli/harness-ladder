from harness_ladder.config import PowerFlags
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM
from harness_ladder.tools import parse_tool_calls, execute_tool_call


def test_loose_xml_parser_recovers_mangled_minicpm_output():
    mangled = 'name="calculator"> name="expression">7 + 5'
    calls = parse_tool_calls(mangled)
    assert calls and calls[0].name == "calculator"
    assert execute_tool_call(calls[0]) == "12"


def test_p5_react_emits_final_answer_after_tool():
    traj = run_v0_loop(
        "calculator 19 + 23 final number",
        task_id="v1_tool_02",
        client=MockLLM(),
        flags=PowerFlags.for_rung(5),
        category="tool",
    )
    assert traj.final_answer == "42"
    assert any("P5 ReAct" in m.content for m in traj.messages if m.role == "system")
    assert any("Final Answer:" in m.content for m in traj.messages if m.role == "assistant")


def test_p5_react_multi_step_long_horizon():
    traj = run_v0_loop(
        "start 5, double, add 3, subtract 4, final?",
        task_id="v1_long_horizon_01",
        client=MockLLM(),
        flags=PowerFlags.for_rung(5),
        category="long_horizon",
    )
    assert traj.final_answer == "9"


def test_p5_allows_multiple_tool_rounds_budget():
    # Smoke: loop completes without hanging when no tools are needed.
    traj = run_v0_loop(
        "Reply with exactly: DONE",
        task_id="adhoc",
        client=MockLLM(),
        flags=PowerFlags.for_rung(5),
    )
    assert traj.final_answer == "DONE"
