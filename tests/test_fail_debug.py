from harness_ladder.config import PowerFlags
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM
from harness_ladder.tools import tools_for_category


def test_file_category_has_no_default_tools():
    assert tools_for_category("file") == ()


def test_math_category_only_calculator():
    names = {s.name for s in tools_for_category("math")}
    assert names == {"calculator"}


def test_long_horizon_fewshot_not_done():
    from harness_ladder.fewshot import pack_few_shot_messages
    msgs = pack_few_shot_messages(category="long_horizon")
    blob = "\n".join(m.content for m in msgs)
    assert "DONE" not in blob
    assert "Z" in blob or "5" in blob


def test_file_task_does_not_call_lookup_under_p3():
    # Mock without tools should answer from... mock returns OK; ensure no lookup in trajectory tools
    traj = run_v0_loop(
        "owners.txt owner Ever, name?",
        task_id="v1_file_03",
        client=MockLLM(),
        flags=PowerFlags.for_rung(8),
        category="file",
    )
    tool_msgs = [m.content for m in traj.messages if m.role == "system" and m.content.startswith("# Tools")]
    assert tool_msgs == []  # file category: no tool schemas injected
    sysblob = "\n".join(m.content for m in traj.messages if m.role == "system")
    assert "read-gate" in sysblob or "Private local reference" in sysblob
