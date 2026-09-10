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

    # a+b, times c, divided by d
    m = re.search(
        r"(\d+)\s*\+\s*(\d+).*times\s+(\d+).*divided by\s+(\d+)",
        text,
        re.I | re.S,
    )
    if m:
        a, b, c, d = m.groups()
        return f"(({a}+{b})*{c})/{d}"

    # visit A then B then C, last — not arithmetic
    if re.search(r"visit\s+\w+\s+then", text, re.I):
        return None
    return None


def is_sequence_task(prompt: str) -> bool:
    """True when the task is not a calculator-forced arithmetic chain."""
    lower = (prompt or "").lower()
    if re.search(r"visit\s+\w+\s+then", lower):
        return True
    if any(k in lower for k in ("reverse ", "first letters", "uppercase", "season after")):
        return True
    return False
