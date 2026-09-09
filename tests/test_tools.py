from harness_ladder.config import PowerFlags
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM
from harness_ladder.tools import (
    execute_tool_call,
    parse_tool_calls,
    render_tool_definitions,
    ToolCall,
)


def test_parse_minicpm_xml_tool_call():
    text = '<function name="calculator"><param name="expression">19+23</param></function>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert calls[0].arguments == {"expression": "19+23"}
    assert execute_tool_call(calls[0]) == "42"


def test_tool_definitions_include_schemas():
    blob = render_tool_definitions()
    assert "# Tools" in blob
    assert "calculator" in blob
    assert "<tools>" in blob


def test_p4_loop_executes_weather_tool():
    traj = run_v0_loop(
        'weather request city Paris unit C, compact JSON',
        task_id="v1_tool_01",
        client=MockLLM(),
        flags=PowerFlags.for_rung(4),
        category="tool",
    )
    assert traj.final_answer == '{"city":"Paris","unit":"C"}'
    assert any("<tool_response>" in m.content for m in traj.messages)


def test_p4_loop_calculator_and_email():
    calc = run_v0_loop(
        "calculator 19 + 23 final number",
        task_id="v1_tool_02",
        client=MockLLM(),
        flags=PowerFlags.for_rung(4),
        category="tool",
    )
    assert calc.final_answer == "42"
    email = run_v0_loop(
        "email recipient ops@example.com, output recipient",
        task_id="v1_tool_04",
        client=MockLLM(),
        flags=PowerFlags.for_rung(4),
        category="tool",
    )
    assert email.final_answer == "ops@example.com"


def test_unknown_tool_returns_error_json():
    out = execute_tool_call(ToolCall(name="nope", arguments={}))
    assert "unknown tool" in out
