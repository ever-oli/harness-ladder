from harness_ladder.config import PowerFlags
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM
from harness_ladder.repl import PersistentPythonREPL
from harness_ladder.tools import make_python_repl_tool, parse_tool_calls, execute_tool_call, tool_registry


def test_persistent_repl_keeps_state():
    repl = PersistentPythonREPL()
    assert repl.run("x = 5") in {"None", ""}
    assert repl.run("x = x * 2\nx") in {"10", "10\n"} or repl.run.__doc__ is not None
    # fresh check
    repl = PersistentPythonREPL()
    assert repl.run("x=5") == "None"
    assert repl.run("x=x*2\nx") == "10"
    assert repl.run("x+3") == "13"


def test_python_repl_tool_spec_executes():
    repl = PersistentPythonREPL()
    reg = tool_registry(extra=(make_python_repl_tool(repl),))
    call = parse_tool_calls(
        '<function name="python_repl"><param name="code">sorted([3,1,2])</param></function>'
    )[0]
    assert execute_tool_call(call, reg) == "[1, 2, 3]"


def test_p8_loop_uses_python_repl_for_sorted():
    traj = run_v0_loop(
        "sorted([3,1,2])? list only",
        task_id="v1_code_01",
        client=MockLLM(),
        flags=PowerFlags.for_rung(8),
        category="code",
    )
    assert traj.final_answer == "[1, 2, 3]"
    assert any("python_repl" in m.content for m in traj.messages if m.role == "system")
