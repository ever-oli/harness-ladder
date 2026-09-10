"""Helpers for plannable long-horizon arithmetic (plan-once / forced tool use)."""

from __future__ import annotations

import re


def suggest_expression(prompt: str) -> str | None:
    """Deterministically map common long-horizon phrasings to one calculator expr."""
    text = (prompt or "").strip().lower()

    # start N, double, add A, subtract B
    m = re.search(
        r"start\s+(\d+).*?double.*?add\s+(\d+).*?subtract\s+(\d+)",
        text,
        re.I | re.S,
    )
    if m:
        n, a, b = m.group(1), m.group(2), m.group(3)
        return f"(({n}*2)+{a})-{b}"

    # start N, double, add A
    m = re.search(r"start\s+(\d+).*?double.*?add\s+(\d+)", text, re.I | re.S)
    if m:
        return f"({m.group(1)}*2)+{m.group(2)}"

    # N packs of M, use K, remain
    m = re.search(r"(\d+)\s+packs?\s+of\s+(\d+).*?(?:use|used)\s+(\d+)", text, re.I | re.S)
    if m:
        return f"({m.group(1)}*{m.group(2)})-{m.group(3)}"

    # add N to a,b,c,d, last result
    m = re.search(
        r"add\s+(\d+)\s+to\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+).*last",
        text,
        re.I | re.S,
    )
    if m:
        add, *nums = m.groups()
        return f"{nums[-1]}+{add}"

    # visit A then B then C, last
    m = re.search(r"visit\s+(\w+)(?:\s+then\s+(\w+))+.*last", text, re.I)
    if m:
        parts = re.findall(r"then\s+(\w+)|visit\s+(\w+)", text, re.I)
        sequence = [a or b for a, b in parts]
        if sequence:
            # Not arithmetic — caller should not force calculator.
            return None
    return None


def is_sequence_task(prompt: str) -> bool:
    return bool(re.search(r"visit\s+\w+\s+then", (prompt or "").lower()))
