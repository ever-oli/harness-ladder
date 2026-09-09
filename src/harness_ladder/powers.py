"""Power flags P0–P12. P0 and P1 are implemented; later powers are stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from harness_ladder.config import ALL_POWERS, POWER_ORDER, PowerFlags


@dataclass(frozen=True)
class PowerSpec:
    id: str
    name: str
    description: str
    implemented: bool = False


POWER_SPECS: Mapping[str, PowerSpec] = {
    "P0": PowerSpec("P0", "V0 sampling loop", "Single-turn sample → answer", True),
    "P1": PowerSpec("P1", "Few-shot exemplars", "Task-family demonstrations before the query", True),
    "P2": PowerSpec("P2", "Budgeted thinking", "Private fixed-budget reasoning with answer-only output", True),
    "P3": PowerSpec("P3", "Self-check", "Verify answer before commit", False),
    "P4": PowerSpec("P4", "Tool schema", "Declare available tools", False),
    "P5": PowerSpec("P5", "Tool use", "Call tools in the loop", False),
    "P6": PowerSpec("P6", "Scratchpad", "Explicit working memory", False),
    "P7": PowerSpec("P7", "Retry / repair", "Retry on failed checks", False),
    "P8": PowerSpec("P8", "Retrieval hook", "Inject retrieved context", False),
    "P9": PowerSpec("P9", "File grounding", "Read task-local files", False),
    "P10": PowerSpec("P10", "Multi-step plan", "Plan then execute", False),
    "P11": PowerSpec("P11", "Reflection", "Post-hoc critique pass", False),
    "P12": PowerSpec("P12", "Recursive ask", "Decompose and recurse", False),
}


def enable_up_to(rung: int) -> PowerFlags:
    """Return flags with P0..P{rung} enabled (cumulative)."""
    return PowerFlags.for_rung(rung)


def list_enabled(flags: PowerFlags) -> list[str]:
    return [p for p in POWER_ORDER if flags.is_on(p)]


def apply_power_hooks(
    flags: PowerFlags,
    *,
    on_stub: Callable[[str], None] | None = None,
) -> list[str]:
    """Return enabled power ids. Unimplemented powers call ``on_stub`` if provided."""
    enabled = list_enabled(flags)
    for pid in enabled:
        spec = POWER_SPECS[pid]
        if not spec.implemented and on_stub is not None:
            on_stub(pid)
    return enabled


assert set(POWER_SPECS) == ALL_POWERS
