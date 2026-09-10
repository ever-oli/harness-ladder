from __future__ import annotations
from collections.abc import Iterable
from harness_ladder.types import Message

# Category exemplars must match the exact-match style of the suite (bare tokens).
# Avoid status-word exemplars for long_horizon (DONE bleed) and file (FILE_OK bleed).
_FEW_SHOT_EXAMPLES: dict[str, tuple[tuple[str, str], ...]] = {
    "echo": (("The requested status word is READY. Give only that word.", "READY"),),
    "math": (
        ("Calculate 3 + 4. Give only the final number.", "7"),
        ("5 pages for 3 days? total pages number only", "15"),
    ),
    "code": (("Evaluate print(2 + 3). Give only the final number.", "5"),),
    "tool": (("calculator 2 + 2 final number", "4"),),
    "file": (
        ("owners.txt says owner = Ada Lovelace, name?", "Ada"),
        ("config.toml port = 9000, what port?", "9000"),
    ),
    "long": (
        ("visit X then Y then Z, last?", "Z"),
        ("start 2, double, add 1, final?", "5"),
    ),
}
_GENERIC_EXAMPLES = (
    ("What is 1 + 1? Give only the final number.", "2"),
    ("Name the color of a clear daytime sky. Give one word.", "blue"),
)
_FAMILY_ALIASES = {
    "echo": "echo",
    "math": "math",
    "arithmetic": "math",
    "code": "code",
    "coding": "code",
    "tool": "tool",
    "tools": "tool",
    "file": "file",
    "grounded": "file",
    "long": "long",
    "long_horizon": "long",
    "horizon": "long",
}


def infer_task_family(
    *, task_id: str = "", category: str | None = None, tags: Iterable[str] | None = None
) -> str | None:
    candidates = ([category] if category else []) + ([str(t) for t in tags] if tags else []) + [task_id]
    for candidate in candidates:
        normalized = str(candidate).lower().replace("-", "_").replace(" ", "_")
        for alias, family in _FAMILY_ALIASES.items():
            if alias in normalized:
                return family
    return None


def pack_few_shot_messages(
    *,
    task_id: str = "",
    category: str | None = None,
    tags: Iterable[str] | None = None,
    max_examples: int = 2,
) -> list[Message]:
    """Return explicit solved user/assistant turns before the final user."""
    if not 1 <= max_examples <= 3:
        raise ValueError(f"max_examples must be in 1..3, got {max_examples}")
    examples = _FEW_SHOT_EXAMPLES.get(
        infer_task_family(task_id=task_id, category=category, tags=tags) or "",
        _GENERIC_EXAMPLES,
    )
    packed: list[Message] = []
    for question, answer in examples[:max_examples]:
        packed.extend((Message(role="user", content=question), Message(role="assistant", content=answer)))
    return packed
