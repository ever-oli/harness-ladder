"""Tests for P0 V0 sampling loop and power flag helpers."""

from harness_ladder.config import ModelConfig, PowerFlags
from harness_ladder.loop import run_v0_loop
from harness_ladder.model import MockLLM
from harness_ladder.powers import POWER_SPECS, enable_up_to, list_enabled


def test_power_flags_rung_cumulative():
    flags = PowerFlags.for_rung(3)
    assert flags.enabled == frozenset({"P0", "P1", "P2", "P3"})
    assert flags.is_on("P0") and flags.is_on("P3")
    assert not flags.is_on("P4")
    assert flags.as_csv() == "P0,P1,P2,P3"


def test_enable_up_to_matches_for_rung():
    assert enable_up_to(0).enabled == frozenset({"P0"})
    assert enable_up_to(12).enabled == frozenset(f"P{i}" for i in range(13))
    assert list_enabled(enable_up_to(2)) == ["P0", "P1", "P2"]


def test_all_power_specs_present():
    assert set(POWER_SPECS) == {f"P{i}" for i in range(13)}
    assert POWER_SPECS["P0"].implemented is True
    assert POWER_SPECS["P1"].implemented is True


def test_v0_loop_exact_reply():
    traj = run_v0_loop(
        "Reply with exactly: OK",
        task_id="t1",
        client=MockLLM(),
        flags=PowerFlags.for_rung(0),
    )
    assert traj.final_answer == "OK"
    assert traj.task_id == "t1"
    assert traj.messages[-1].role == "assistant"
    assert traj.metadata["powers"] == "P0"


def test_v0_loop_math_and_higher_rung_stub_safe():
    config = ModelConfig(seed=42)
    traj = run_v0_loop(
        "What is 2 + 2? Reply with only the number.",
        client=MockLLM(config),
        config=config,
        flags=PowerFlags.for_rung(5),
    )
    assert traj.final_answer == "4"
    assert "P0" in traj.metadata["powers"]
    assert "P5" in traj.metadata["powers"]


def test_p0_stays_bare_loop():
    traj = run_v0_loop(
        "Reply with exactly: OK",
        client=MockLLM(),
        flags=PowerFlags.for_rung(0),
    )
    assert [(m.role, m.content) for m in traj.messages] == [
        ("system", "You are a helpful assistant. Follow instructions precisely."),
        ("user", "Reply with exactly: OK"),
        ("assistant", "OK"),
    ]

def test_p1_packs_family_aware_exemplar_before_query():
    traj = run_v0_loop(
        "What is 2 + 2? Reply with only the number.",
        task_id="smoke_math_1",
        category="math",
        client=MockLLM(),
        flags=PowerFlags.for_rung(1),
    )
    assert traj.final_answer == "4"
    assert [m.role for m in traj.messages] == ["system", "user", "assistant", "user", "assistant"]
    assert traj.messages[1].content == "Calculate 3 + 4. Give only the final number."
    assert traj.messages[2].content == "7"
    assert traj.messages[3].content.endswith("only the number.")

def test_p1_uses_generic_answer_only_examples_without_metadata():
    traj = run_v0_loop(
        "Return the requested value.",
        client=MockLLM(),
        flags=PowerFlags.for_rung(1),
    )
    assert len(traj.messages) == 7
    assert traj.messages[1].role == "user"
    assert traj.messages[2].role == "assistant"
    assert traj.messages[2].content == "2"
