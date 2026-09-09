"""P8 — persistent, restricted Python REPL for tool-using agents."""

from __future__ import annotations

import ast
import io
import traceback
from contextlib import redirect_stdout
from typing import Any


_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class PersistentPythonREPL:
    """Stateful restricted Python REPL used by the P8 ``python_repl`` tool."""

    def __init__(self) -> None:
        self._ns: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}

    def reset(self) -> None:
        self._ns = {"__builtins__": dict(_SAFE_BUILTINS)}

    def run(self, code: str, *, max_chars: int = 2000) -> str:
        source = (code or "").strip()
        if not source:
            return ""
        buf = io.StringIO()
        try:
            tree = ast.parse(source, mode="exec")
            body = list(tree.body)
            last_expr: ast.Expr | None = None
            if body and isinstance(body[-1], ast.Expr):
                last_expr = body.pop()  # type: ignore[assignment]
            with redirect_stdout(buf):
                if body:
                    exec(
                        compile(ast.Module(body=body, type_ignores=[]), "<repl>", "exec"),
                        self._ns,
                        self._ns,
                    )
                value = None
                if last_expr is not None:
                    value = eval(
                        compile(ast.Expression(last_expr.value), "<repl>", "eval"),
                        self._ns,
                        self._ns,
                    )
            printed = buf.getvalue()
            chunks: list[str] = []
            if printed:
                chunks.append(printed.rstrip("\n"))
            if last_expr is not None:
                chunks.append(value if isinstance(value, str) else repr(value))
            return ("\n".join(chunks) if chunks else "None")[:max_chars]
        except Exception:  # noqa: BLE001
            return f"Error: {traceback.format_exc(limit=1).strip()}"[:max_chars]
