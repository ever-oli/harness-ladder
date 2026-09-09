"""P0 — V0 sampling loop with cumulative P1–P8 harness powers."""

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
from harness_ladder.repl import PersistentPythonREPL
from harness_ladder.tools import (
    execute_tool_call,
    format_tool_response,
    infer_tool_hint,
    make_python_repl_tool,
    parse_tool_calls,
    render_tool_definitions,
    tool_registry,
)
from harness_ladder.refine import self_refine
from harness_ladder.reflexion import reflect, retry_with_reflection
from harness_ladder.types import Message, Trajectory

_P2_PROTOCOL = (
    "P2: reason silently for at most {budget} internal tokens; emit only the final answer. "
    "Never output scratch work, chain-of-thought, or <think> tags."
)

_P5_REACT = (
    "P5 ReAct protocol (budget ~{budget} tokens of private thought per step):\n"
    "1. Optionally write one short line: Thought: <plan>\n"
    "2. Either call a tool with MiniCPM XML "
    '<function name="tool"><param name="k">v</param></function>\n'
    "   or finish with: Final Answer: <answer only>\n"
    "3. After a <tool_response>, think again or give Final Answer.\n"
    "Do not invent tool results. Prefer tools for arithmetic, weather, lookup, email, search."
)

_FINAL_RE = re.compile(r"(?im)^\s*Final Answer:\s*(.+)\s*$")
_THOUGHT_LINE_RE = re.compile(r"(?im)^\s*Thought:\s*.+$")


def _p2_answer_only(text: str) -> str:
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"<think\b[^>]*>.*$", "", text, flags=re.I | re.S)
    return text.split("FINAL_ANSWER:", 1)[-1].strip()


def _extract_final_answer(text: str) -> str | None:
    matches = list(_FINAL_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _finalize_answer(text: str, flags: PowerFlags) -> str:
    answer = (text or "").strip()
    if flags.is_on("P5"):
        extracted = _extract_final_answer(answer)
        if extracted:
            answer = extracted
        else:
            # Drop Thought lines; keep last non-empty content line.
            lines = [
                line.strip()
                for line in answer.splitlines()
                if line.strip() and not _THOUGHT_LINE_RE.match(line)
            ]
            if lines:
                answer = lines[-1]
    if flags.is_on("P2") and not flags.is_on("P5"):
        answer = _p2_answer_only(answer)
    if flags.is_on("P4"):
        calls = parse_tool_calls(answer)
        if calls:
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
    max_tool_rounds: int | None = None,
) -> Trajectory:
    """Execute the sampling loop with cumulative P0–P8 powers.

    P4: tool definitions + one call round.
    P5: ReAct multi-step thought → act → observe (default up to 3 tool rounds).
    P6: self-refine (critique → compact revise) after the draft answer.
    P7: Reflexion — verbal critique, then one retry trial with that memory.
    P8: persistent Python REPL tool (stateful across tool rounds).
    """
    config = config or ModelConfig()
    flags = flags or PowerFlags.for_rung(0)
    client = client or MockLLM(config)

    apply_power_hooks(flags)

    messages = [Message(role="system", content=system)]
    if flags.is_on("P5"):
        messages.append(
            Message(role="system", content=_P5_REACT.format(budget=config.reasoning_token_budget))
        )
    elif flags.is_on("P2"):
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
    repl = PersistentPythonREPL() if flags.is_on("P8") else None
    extra_tools = (make_python_repl_tool(repl),) if repl is not None else None
    if flags.is_on("P4"):
        tool_msg = render_tool_definitions(extra=extra_tools)
        hint = infer_tool_hint(prompt, python_repl=flags.is_on("P8"))
        if hint:
            tool_msg = tool_msg + "\n\n" + hint
        if flags.is_on("P8"):
            tool_msg += (
                "\n\nP8: python_repl is persistent within this task. "
                "Use it for code execution; Final Answer with the printed/returned value only."
            )
        messages.append(Message(role="system", content=tool_msg))

    messages.append(Message(role="user", content=prompt))

    t0 = time.perf_counter()
    answer = ""
    registry = tool_registry(extra=extra_tools) if flags.is_on("P4") else {}
    if max_tool_rounds is None:
        if flags.is_on("P8") or flags.is_on("P5"):
            rounds = 4 if flags.is_on("P8") else 3
        elif flags.is_on("P4"):
            rounds = 1
        else:
            rounds = 0
    else:
        rounds = max_tool_rounds if flags.is_on("P4") else 0

    for _ in range(rounds + 1):
        raw = client.complete(messages).strip()
        messages.append(Message(role="assistant", content=raw))

        if flags.is_on("P5") and _extract_final_answer(raw) is not None:
            answer = raw
            break

        if flags.is_on("P4"):
            calls = parse_tool_calls(raw)
            if calls and rounds > 0:
                # Execute all parsed calls this turn (usually one); feed observations.
                observations = []
                for call in calls[:3]:
                    observations.append(execute_tool_call(call, registry))
                obs = "\n".join(format_tool_response(item) for item in observations)
                if flags.is_on("P5"):
                    obs += "\nContinue ReAct. Use Final Answer: when done."
                messages.append(Message(role="user", content=obs))
                rounds -= 1
                continue
        answer = raw
        break
    else:
        answer = messages[-1].content if messages else ""

    answer = _finalize_answer(answer, flags)
    if flags.is_on("P6"):
        refined = self_refine(client, prompt, answer)
        if refined != answer:
            messages.append(Message(role="assistant", content=f"P6_REFINED: {refined}"))
        answer = refined
    if flags.is_on("P7"):
        reflection = reflect(client, prompt, answer)
        messages.append(Message(role="assistant", content=f"P7_REFLECTION: {reflection}"))
        retried = retry_with_reflection(client, prompt, answer, reflection)
        retried = _finalize_answer(retried, flags)
        if flags.is_on("P6"):
            retried = self_refine(client, prompt, retried)
        messages.append(Message(role="assistant", content=f"P7_RETRY: {retried}"))
        # Prefer retry when non-empty; Reflexion is a second trial.
        if retried:
            answer = retried
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
