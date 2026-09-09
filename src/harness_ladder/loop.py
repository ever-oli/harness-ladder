"""P0 — V0 sampling loop: one user prompt → one model completion → final answer."""

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
from harness_ladder.types import Message, Trajectory

_P2_PROTOCOL = "P2: reason silently for at most {budget} internal tokens; emit only the final answer. Never output scratch work, chain-of-thought, or <think> tags."
def _p2_answer_only(text):
    text=re.sub(r"<think\b[^>]*>.*?</think>","",text,flags=re.I|re.S)
    text=re.sub(r"<think\b[^>]*>.*$","",text,flags=re.I|re.S)
    return text.split("FINAL_ANSWER:",1)[-1].strip()


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
) -> Trajectory:
    """Execute the sampling loop, optionally adding P1 few-shot context.

    With P0 only, this remains the original bare system+query loop. When P1
    is enabled, solved user/assistant exemplars are inserted immediately before
    the query; the control flow and completion call are otherwise unchanged.
    """
    config = config or ModelConfig()
    flags = flags or PowerFlags.for_rung(0)
    client = client or MockLLM(config)

    apply_power_hooks(flags)  # later powers remain no-ops until implemented

    messages = [Message(role="system", content=system)]
    if flags.is_on("P2"):
        messages.append(Message(role="system", content=_P2_PROTOCOL.format(budget=config.reasoning_token_budget)))
    if flags.is_on("P1"):
        messages.extend(
            pack_few_shot_messages(task_id=task_id, category=category, tags=tags)
        )
    if flags.is_on("P3"):
        corpus = Path(__file__).resolve().parents[2] / "corpus" / "task_passages.md"
        context = render_context(retrieve_from_path(prompt, corpus, top_k=3))
        if context:
            messages.append(Message(role="system", content=context))

    messages.append(Message(role="user", content=prompt))

    t0 = time.perf_counter()
    answer = client.complete(messages).strip()
    if flags.is_on("P2"):
        answer = _p2_answer_only(answer)
    elapsed = time.perf_counter() - t0

    messages.append(Message(role="assistant", content=answer))

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
