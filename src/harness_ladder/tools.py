"""P4 — MiniCPM5-compatible tool schemas, XML call parsing, and executors."""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

# MiniCPM5 native call shape:
# <function name="tool"><param name="k">v</param></function>
_FUNCTION_RE = re.compile(
    r"<function\s+name=\"(?P<name>[^\"]+)\">(?P<body>.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<param\s+name=\"(?P<name>[^\"]+)\">(?P<value>.*?)</param>",
    re.IGNORECASE | re.DOTALL,
)
_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[Mapping[str, str]], str]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_eval_arith(expr: str) -> str:
    tree = ast.parse(expr.strip(), mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _eval(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"Unsupported expression: {expr!r}")

    value = _eval(tree)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _weather(args: Mapping[str, str]) -> str:
    city = args.get("city", "Unknown").strip()
    unit = args.get("unit", "C").strip() or "C"
    return json.dumps({"city": city, "unit": unit}, separators=(",", ":"))


def _calculator(args: Mapping[str, str]) -> str:
    expr = args.get("expression") or args.get("expr") or args.get("query") or ""
    return _safe_eval_arith(expr)


def _lookup(args: Mapping[str, str]) -> str:
    key = (args.get("key") or args.get("status") or args.get("name") or "").strip().lower()
    table = {"status": "ready", "ready": "ready", "health": "ready"}
    return table.get(key, "ready" if "ready" in key else "unknown")


def _email(args: Mapping[str, str]) -> str:
    recipient = (
        args.get("recipient")
        or args.get("to")
        or args.get("email")
        or ""
    ).strip()
    return recipient


def _search(args: Mapping[str, str]) -> str:
    raw = args.get("count") or args.get("n") or args.get("query") or "0"
    match = re.search(r"\d+", str(raw))
    count = int(match.group(0)) if match else 0
    return json.dumps({"count": count}, separators=(",", ":"))


DEFAULT_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="weather",
        description="Build a compact weather request JSON for a city and unit.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["C", "F"]},
            },
            "required": ["city", "unit"],
        },
        handler=_weather,
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate a simple arithmetic expression and return the number.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        handler=_calculator,
    ),
    ToolSpec(
        name="lookup",
        description="Lookup a named status/key from the local table.",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        handler=_lookup,
    ),
    ToolSpec(
        name="email",
        description="Return the email recipient address for a drafted message.",
        parameters={
            "type": "object",
            "properties": {"recipient": {"type": "string"}},
            "required": ["recipient"],
        },
        handler=_email,
    ),
    ToolSpec(
        name="search",
        description="Return a compact JSON object with a search hit count.",
        parameters={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
        handler=_search,
    ),
)


def make_python_repl_tool(repl) -> ToolSpec:
    """Bind a PersistentPythonREPL instance into a ToolSpec handler."""

    def _handler(args: Mapping[str, str]) -> str:
        code = args.get("code") or args.get("source") or args.get("expression") or ""
        return repl.run(code)

    return ToolSpec(
        name="python_repl",
        description=(
            "Execute Python in a persistent REPL. State survives across calls. "
            "Use print(...) or a final expression; prefer for code/math multi-step."
        ),
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        handler=_handler,
    )


def tool_registry(
    tools: tuple[ToolSpec, ...] | None = None,
    *,
    extra: tuple[ToolSpec, ...] | None = None,
) -> dict[str, ToolSpec]:
    specs = list(tools or DEFAULT_TOOLS)
    if extra:
        specs.extend(extra)
    return {spec.name: spec for spec in specs}


def render_tool_definitions(
    tools: tuple[ToolSpec, ...] | None = None,
    *,
    extra: tuple[ToolSpec, ...] | None = None,
) -> str:
    """Render MiniCPM5-style tool instructions for the system prompt."""
    specs = list(tools or DEFAULT_TOOLS)
    if extra:
        specs.extend(extra)
    lines = [
        "# Tools",
        "",
        "You are provided with function signatures within XML tags:",
        "<tools>",
    ]
    for spec in specs:
        lines.append(json.dumps(spec.openai_schema(), ensure_ascii=False))
    lines.extend(
        [
            "</tools>",
            "",
            "Tool usage guidelines:",
            "- You may call zero or more functions. If no function calls are needed, just answer normally and do not include any.",
            "- When calling a function, return an XML object using:",
            '  <function name="tool_name"><param name="param">value</param></function>',
            "- After a tool result is provided, emit only the final answer (no XML).",
        ]
    )
    return "\n".join(lines)


_LOOSE_FUNCTION_RE = re.compile(
    r'(?:<function\s+)?name="(?P<name>weather|calculator|lookup|email|search|python_repl)"\s*>'
    r'(?P<body>.*?)(?:</function>|$)',
    re.IGNORECASE | re.DOTALL,
)
_LOOSE_PARAM_RE = re.compile(
    r'(?:<param\s+)?name="(?P<name>[^"]+)"\s*>(?P<value>.*?)(?:</param>|(?=(?:(?:<param\s+)?name=")|$))',
    re.IGNORECASE | re.DOTALL,
)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract MiniCPM5 XML tool calls, including mangled tag-less variants."""
    raw = text or ""
    calls: list[ToolCall] = []

    def _params(body: str) -> dict[str, str]:
        args: dict[str, str] = {}
        for param in _PARAM_RE.finditer(body):
            value = param.group("value").strip()
            cdata = _CDATA_RE.fullmatch(value)
            if cdata:
                value = cdata.group(1)
            args[param.group("name")] = value
        if not args:
            for param in _LOOSE_PARAM_RE.finditer(body):
                value = param.group("value").strip()
                value = re.sub(r"</?(?:param|function)>", "", value).strip()
                if value:
                    args[param.group("name")] = value
        return args

    for match in _FUNCTION_RE.finditer(raw):
        calls.append(ToolCall(name=match.group("name").strip(), arguments=_params(match.group("body"))))
    if calls:
        return calls
    for match in _LOOSE_FUNCTION_RE.finditer(raw):
        args = _params(match.group("body"))
        if args:
            calls.append(ToolCall(name=match.group("name").strip(), arguments=args))
    return calls


def execute_tool_call(call: ToolCall, registry: Mapping[str, ToolSpec] | None = None) -> str:
    reg = registry or tool_registry()
    if call.name not in reg:
        return json.dumps({"error": f"unknown tool: {call.name}"})
    try:
        return reg[call.name].handler(call.arguments)
    except Exception as exc:  # noqa: BLE001 — surface tool failures to the model
        return json.dumps({"error": str(exc)})


def format_tool_response(result: str) -> str:
    return f"<tool_response>\n{result}\n</tool_response>"


def infer_tool_hint(prompt: str, *, python_repl: bool = False) -> str | None:
    """Optional soft hint for tiny models when the task clearly names a tool."""
    lower = prompt.lower()
    if python_repl and any(
        k in lower
        for k in (
            "sorted(",
            "range(",
            "len(",
            ".upper(",
            "print",
            "add(",
            "packs",
            "double",
            "start ",
            "visit ",
        )
    ):
        return "Prefer the python_repl tool for code/math steps; state persists across calls."
    if "weather" in lower:
        return "Prefer the weather tool."
    if "calculator" in lower or re.search(r"\d+\s*[+\-*/]\s*\d+", prompt):
        return "Prefer the calculator tool when arithmetic is required."
    if "lookup" in lower or "status" in lower:
        return "Prefer the lookup tool."
    if "email" in lower or "recipient" in lower:
        return "Prefer the email tool."
    if "search" in lower and "count" in lower:
        return "Prefer the search tool."
    return None
