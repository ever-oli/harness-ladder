"""P7 — Reflexion: verbal self-critique across a second trial."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from harness_ladder.types import Message

if TYPE_CHECKING:
    from harness_ladder.model import LLMClient

_REFLEXION_MARKER = "P7_REFLEXION"


def build_reflection_messages(prompt: str, draft: str) -> list[Message]:
    body = (
        f"{_REFLEXION_MARKER}\n"
        "Write a short Reflection (1-2 sentences) about mistakes in the draft "
        "for exact-match grading.\n"
        "Rules: prefer bare numbers/tokens; NEVER suggest adding units, currency signs, "
        "or restating the question. Flag verbosity and wrong values only.\n"
        "Do not give the final answer yet.\n\n"
        f"Question: {prompt}\n"
        f"Draft: {draft}\n"
        "Reflection:"
    )
    return [
        Message(role="system", content="You write brief Reflexion-style critiques."),
        Message(role="user", content=body),
    ]


def build_retry_messages(prompt: str, draft: str, reflection: str) -> list[Message]:
    body = (
        f"{_REFLEXION_MARKER}_RETRY\n"
        "Use the reflection to answer again. Output ONLY the bare exact final answer (no $, no units, no sentence).\n\n"
        f"Question: {prompt}\n"
        f"Previous draft: {draft}\n"
        f"Reflection: {reflection}\n"
        "Final Answer:"
    )
    return [
        Message(role="system", content="You retry after Reflexion. Answer only."),
        Message(role="user", content=body),
    ]


def reflect(client: "LLMClient", prompt: str, draft: str) -> str:
    raw = client.complete(build_reflection_messages(prompt, draft)).strip()
    raw = re.sub(r"(?im)^\s*Reflection:\s*", "", raw).strip()
    return raw.splitlines()[0].strip() if raw else "Be precise and brief."


def retry_with_reflection(client: "LLMClient", prompt: str, draft: str, reflection: str) -> str:
    raw = client.complete(build_retry_messages(prompt, draft, reflection)).strip()
    raw = re.sub(r"(?im)^\s*Final Answer:\s*", "", raw).strip()
    return raw
