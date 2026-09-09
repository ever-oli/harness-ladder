"""P0 — V0 sampling loop with cumulative P1–P4 harness powers."""

from __future__ import annotations

import re
from pathlib import Path
import time
from collections.abc import Iterable
from typing import Optional

from harness_ladder.config import ModelConfig, PowerFlags
from harness_ladder.model import LLMClient, MockLLM
from harness_ladder.fewshot import pack_few_shot_messages
from harness_ladder.retriever import render_context, retrieve_from_path
from harness_ladder.powers import apply_power_hooks
from harness_ladder.tools import (
    execute_tool_call,
    format_tool_response,
    infer_tool_hint,
    parse_tool_calls,
    render_tool_definitions,
    tool_registry,
)
from harness_ladder.types import Message, Trajectory

_P2_PROTOCOL = (
    "P2: reason silently for at most {budget} internal tokens; emit only the final answer. "
    "Never output scratch work, chain-of-thought, or <think> tags."
)


def _p2_answer_only(text: str) -> str:
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"<think\b[^>]*>.*$", "", text, flags=re.I | re.S)
    return text.split("FINAL_ANSWER:", 1)[-1].strip()


def _finalize_answer(text: str, flags: PowerFlags) -> str:
    answer = text.strip()
    if flags.is_on("P2"):
        answer = _p2_answer_only(answer)
    # Strip leftover tool XML if the model mixed formats.
    if flags.is_on("P4") and "<function" in answer.lower():
        calls = parse_tool_calls(answer)
        if calls:
            # Prefer executed tool result when the model never produced a final turn.
            answer = execute_tool_call(calls[0])
    return answer.strip()


def run_v0_loop(
    prompt: str,
    *,
    task_id: str = "adhoc",
    client: Optional[LLMClient] = None,
    config: Optional[ModelConfig] = None,
    flags: Optional[PowerFlags] = None,
    system: str = "You are a helpful assistant. Follow instructions precisely.",
    category: str | None = None,
    tags: Iterable[str] | None = None,
    max_tool_rounds: int = 1,
) -> Trajectory:
    """Execute the sampling loop with cumulative P0–P4 powers.

    P4 adds MiniCPM5-style tool definitions and a single tool-call round
    (call → observe → final answer). Multi-step ReAct belongs to P5.
    """
    config = config or ModelConfig()
    flags = flags or PowerFlags.for_rung(0)
    client = client or MockLLM(config)

    apply_power_hooks(flags)

    messages = [Message(role="system", content=system)]
    if flags.is_on("P2"):
        messages.append(
            Message(role="system", content=_P2_PROTOCOL.format(budget=config.reasoning_token_budget))
        )
    if flags.is_on("P1"):
        messages.extend(pack_few_shot_messages(task_id=task_id, category=category, tags=tags))
    if flags.is_on("P3"):
        corpus = Path(__file__).resolve().parents[2] / "corpus" / "task_passages.md"
        context = render_context(retrieve_from_path(prompt, corpus, top_k=3))
        if context:
            messages.append(Message(role="system", content=context))
    if flags.is_on("P4"):
        tool_msg = render_tool_definitions()
        hint = infer_tool_hint(prompt)
        if hint:
            tool_msg = tool_msg + "\n\n" + hint
        messages.append(Message(role="system", content=tool_msg))

    messages.append(Message(role="user", content=prompt))

    t0 = time.perf_counter()
    answer = ""
    registry = tool_registry() if flags.is_on("P4") else {}
    rounds = max_tool_rounds if flags.is_on("P4") else 0

    for _ in range(rounds + 1):
        raw = client.complete(messages).strip()
        messages.append(Message(role="assistant", content=raw))
        if flags.is_on("P4"):
            calls = parse_tool_calls(raw)
            if calls and rounds > 0:
                # Execute the first call only (P4 = definitions + calling, not multi-act ReAct).
                result = execute_tool_call(calls[0], registry)
                messages.append(Message(role="user", content=format_tool_response(result)))
                rounds -= 1
                continue
        answer = raw
        break
    else:
        answer = messages[-1].content if messages else ""

    answer = _finalize_answer(answer, flags)
    elapsed = time.perf_counter() - t0

    return Trajectory(
        task_id=task_id,
        messages=messages,
        final_answer=answer,
        tokens_used=max(1, len(answer.split())),
        wall_time_s=elapsed,
        metadata={
            "model_id": config.model_id,
            "powers": flags.as_csv(),
            "rung": max((int(p[1:]) for p in flags.enabled), default=0),
            "seed": config.seed,
        },
    )
