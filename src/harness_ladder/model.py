"""LLM clients: MockLLM for CI and an OpenAI-compatible stub for live serving."""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from harness_ladder.config import ModelConfig
from harness_ladder.types import Message


def clean_completion(text: str) -> str:
    """Remove model-only control/thinking markup before exact-match grading."""
    text = re.sub(r"<\|[^>]*\|>", "", text)
    if "<think>" in text.lower():
        if re.search(r"</think>", text, flags=re.IGNORECASE):
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        else:
            body = re.split(r"<think>", text, maxsplit=1, flags=re.IGNORECASE)[-1]
            candidates = re.findall(r"(?:[A-Z][A-Z_]+|\d+)(?=[^A-Za-z0-9_]*$)", body.strip())
            text = candidates[-1] if candidates else body.splitlines()[-1]
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1] if lines else "").strip("`*_ \t")


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message]) -> str: ...


class MockLLM:
    """Deterministic, offline mock for CI and local smoke tests.

    Behavior:
    - If the last user message contains ``Reply with exactly: X``, return X.
    - Else if it asks for a simple arithmetic expression, evaluate it.
    - Else echo a short fallback so the loop always completes.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def complete(self, messages: Sequence[Message]) -> str:
        user_text = ""
        for m in reversed(messages):
            if m.role == "user":
                user_text = m.content
                break

        # After a tool response, emit the tool payload as the final answer.
        if user_text.strip().startswith("<tool_response>"):
            body = re.sub(r"</?tool_response>", "", user_text).strip()
            return body

        system_blob = "\n".join(m.content for m in messages if m.role == "system")
        tools_enabled = "# Tools" in system_blob or "<tools>" in system_blob

        exact = re.search(r"Reply with exactly:\s*(.+)$", user_text, re.IGNORECASE | re.MULTILINE)
        if exact:
            return exact.group(1).strip()

        # "What is A + B?" / "What is A - B?" style
        arith = re.search(
            r"What is\s+(\d+)\s*([+\-*/])\s*(\d+)\??",
            user_text,
            re.IGNORECASE,
        )
        if arith:
            a, op, b = int(arith.group(1)), arith.group(2), int(arith.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else 0}
            return str(ops[op])

        # print(1+1) style
        code = re.search(r"print\((\d+)\s*\+\s*(\d+)\)", user_text)
        if code:
            return str(int(code.group(1)) + int(code.group(2)))

        if tools_enabled:
            lower = user_text.lower()
            if "weather" in lower and "paris" in lower:
                return '<function name="weather"><param name="city">Paris</param><param name="unit">C</param></function>'
            calc = re.search(r"calculator\s+(\d+)\s*\+\s*(\d+)", user_text, re.I)
            if calc:
                expr = f"{calc.group(1)}+{calc.group(2)}"
                return f'<function name="calculator"><param name="expression">{expr}</param></function>'
            if "lookup" in lower or "status" in lower:
                return '<function name="lookup"><param name="key">status</param></function>'
            email = re.search(r"([\w.+-]+@[\w.-]+)", user_text)
            if "email" in lower and email:
                return f'<function name="email"><param name="recipient">{email.group(1)}</param></function>'
            if "search" in lower and "count" in lower:
                return '<function name="search"><param name="count">3</param></function>'

        return "OK"


class OpenAICompatibleClient:
    """Thin OpenAI-compatible chat client (vLLM / OpenAI-style base_url).

    Not used in CI. Requires a live server when called.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    def complete(self, messages: Sequence[Message]) -> str:
        client = self._ensure_client()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = client.chat.completions.create(
            model=self.config.model_id,
            messages=payload,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed,
        )
        return clean_completion(resp.choices[0].message.content or "")

import time


class TransformersLLM:
    """Lazy local Transformers adapter with deterministic MiniCPM-safe generation."""
    def __init__(self, config: ModelConfig | None = None, *, model_id: str | None = None, max_new_tokens: int | None = None) -> None:
        self.config = config or ModelConfig()
        self.model_id = model_id or self.config.model_id
        self.max_new_tokens = max_new_tokens or self.config.max_tokens
        self.last_metadata = {}
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        kw = {"torch_dtype": torch.float16 if cuda else torch.float32, "trust_remote_code": True}
        if cuda:
            kw["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
        if not cuda:
            self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def clean_generation_inputs(inputs):
        inputs.pop("token_type_ids", None)
        return inputs

    def complete(self, messages: Sequence[Message]) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            prompt = self.tokenizer.apply_chat_template(
                payload,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizers may not accept enable_thinking.
            prompt = self.tokenizer.apply_chat_template(
                payload,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.clean_generation_inputs(
            self.tokenizer(prompt, return_tensors="pt")
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_prompt = int(inputs["input_ids"].shape[-1])
        t0 = time.perf_counter()
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0
        seq = output[0]
        ids = inputs["input_ids"][0]
        if seq.shape[-1] >= n_prompt and self._torch.equal(seq[:n_prompt].cpu(), ids.cpu()):
            seq = seq[n_prompt:]
        text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
        self.last_metadata = {
            "prompt_tokens": n_prompt,
            "output_tokens": int(seq.numel()),
            "latency_s": latency,
            "max_new_tokens": self.max_new_tokens,
            "device": str(self.model.device),
            "dtype": str(next(self.model.parameters()).dtype),
        }
        # Keep raw tool XML for the P4 loop; only strip think markup.
        if "<function" in text.lower():
            text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S)
            text = re.sub(r"<think\b[^>]*>.*$", "", text, flags=re.I | re.S)
            return text.strip()
        return clean_completion(text)
