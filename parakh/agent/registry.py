"""Tool registry: @tool registers a function, call_tool validates args and runs it.

    @tool(
        name="my_tool",
        description="What it does and when to use it.",
        parameters={
            "query": {"type": "string", "description": "..."},
            "limit": {"type": "integer", "description": "...", "minimum": 1, "maximum": 20},
        },
        required=["query"],
    )
    def my_tool(query, limit=5):
        return {...}          # anything JSON-serialisable
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    required: list[str]
    func: Callable[..., Any] = field(repr=False)


TOOLS: dict[str, Tool] = {}


def tool(
    name: str,
    description: str,
    parameters: dict[str, dict[str, Any]],
    required: list[str] | None = None,
):
    """Register a function as an agent tool."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        TOOLS[name] = Tool(name, description, parameters, required or [], func)
        return func

    return decorator


def describe_tools() -> str:
    """Render the registry as text for the LLM prompt."""
    lines = []
    for t in TOOLS.values():
        args = [
            f'    - {arg} ({spec.get("type")}, {"required" if arg in t.required else "optional"}): '
            f'{spec.get("description", "")}'
            for arg, spec in t.parameters.items()
        ]
        lines.append(f"* {t.name}: {t.description}\n" + "\n".join(args))
    return "\n".join(lines)


def _coerce(arg: str, value: Any, spec: dict[str, Any]) -> Any:
    kind = spec.get("type")
    if kind == "integer":
        if isinstance(value, bool):
            raise ValueError(f"{arg} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{arg} must be an integer") from None
        low, high = spec.get("minimum"), spec.get("maximum")
        if low is not None:
            number = max(number, low)
        if high is not None:
            number = min(number, high)
        return number
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{arg} must be a non-empty string")
        return value.strip()
    if kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{arg} must be a list of strings")
        return [str(v).strip() for v in value if str(v).strip()]
    return value


def call_tool(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Validate arguments and run a tool. Never raises; returns ok/result/error."""
    registered = TOOLS.get(name)
    if registered is None:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {sorted(TOOLS)}"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "args must be an object"}
    unknown = set(args) - set(registered.parameters)
    if unknown:
        return {"ok": False, "error": f"unknown arguments {sorted(unknown)}"}
    missing = [a for a in registered.required if args.get(a) in (None, "")]
    if missing:
        return {"ok": False, "error": f"missing required arguments {missing}"}
    try:
        clean = {
            arg: _coerce(arg, value, registered.parameters[arg])
            for arg, value in args.items()
            if value is not None
        }
        return {"ok": True, "result": registered.func(**clean)}
    except Exception as error:
        LOGGER.warning("tool %s failed: %s", name, error)
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
