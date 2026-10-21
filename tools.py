"""Tool definitions for the master agent.

Every tool the agent can call lives here. To add one, write a function and
decorate it with @tool(...); the master agent picks it up automatically (its
name, description and parameters are shown to the LLM, and call_tool() validates
the arguments before running it).

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

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

LOGGER = logging.getLogger("tools")

RRF_K = 60  # reciprocal-rank-fusion constant


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
        args = []
        for arg, spec in t.parameters.items():
            marker = "required" if arg in t.required else "optional"
            args.append(
                f'    - {arg} ({spec.get("type")}, {marker}): {spec.get("description", "")}'
            )
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
        if low is not None and number < low:
            number = low
        if high is not None and number > high:
            number = high
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


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def _default_retriever(query: str, form_id: int | None, top_k: int) -> list[dict[str, Any]]:
    """Existing vector search, imported lazily so tools load without models/DB."""
    from query_resolver import retrieve_top_k

    return retrieve_top_k(query, form_id=form_id, top_k=top_k)


RETRIEVER: Callable[[str, int | None, int], list[dict[str, Any]]] = _default_retriever


def _db_rows(query: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    from db_conn import execute

    return json.loads(execute(query, params))


@tool(
    name="search_questions",
    description=(
        "Semantic search over Parakh inspection-form questions (English + Hindi). "
        "Use for any request to find which form fields/questions relate to a "
        "topic. Pass a clear English query plus 2-4 variants mixing English and "
        "Hindi, worded the way a form field would be. Returns ranked matches."
    ),
    parameters={
        "query": {"type": "string", "description": "Self-contained English search sentence."},
        "variants": {
            "type": "array",
            "description": "Alternative phrasings (English and Hindi/Devanagari).",
        },
        "top_k": {
            "type": "integer",
            "description": "How many matches to return (default 4).",
            "minimum": 1,
            "maximum": 20,
        },
        "form_id": {"type": "integer", "description": "Restrict to one form id.", "minimum": 1},
    },
    required=["query"],
)
def search_questions(
    query: str,
    variants: list[str] | None = None,
    top_k: int = 4,
    form_id: int | None = None,
) -> list[dict[str, Any]]:
    """Search every phrasing, fuse by reciprocal rank, dedupe by (form_id, ques_id)."""
    seen_text: set[str] = set()
    phrasings = []
    for text in [query, *(variants or [])][:5]:
        if text.lower() not in seen_text:
            seen_text.add(text.lower())
            phrasings.append(text)

    fused: dict[tuple[Any, Any], dict[str, Any]] = {}
    for phrasing in phrasings:
        try:
            rows = RETRIEVER(phrasing, form_id, max(top_k * 2, 10))
        except Exception as error:
            LOGGER.warning("retrieval failed for %r: %s", phrasing, error)
            continue
        rank = 0
        counted: set[tuple[Any, Any]] = set()
        for row in rows:  # best-first; en/hi rows can repeat one question
            key = (row.get("form_id"), row.get("ques_id"))
            if key in counted:
                continue
            counted.add(key)
            rank += 1
            item = fused.setdefault(
                key,
                {
                    "form_id": row.get("form_id"),
                    "ques_id": row.get("ques_id"),
                    "question_caption": row.get("question_caption"),
                    "question_caption_hindi": row.get("question_caption_hindi"),
                    "rrf_score": 0.0,
                    "best_similarity": 0.0,
                    "matched_queries": 0,
                },
            )
            item["rrf_score"] += 1.0 / (RRF_K + rank)
            item["matched_queries"] += 1
            if row.get("similarity") is not None:
                item["best_similarity"] = max(item["best_similarity"], float(row["similarity"]))
    ranked = sorted(fused.values(), key=lambda r: (-r["rrf_score"], -r["best_similarity"]))[:top_k]
    for item in ranked:
        item["rrf_score"] = round(item["rrf_score"], 4)
        item["best_similarity"] = round(item["best_similarity"], 3)
    return ranked


@tool(
    name="get_form_questions",
    description="List all questions of one form in order. Use when the user asks what a form contains.",
    parameters={
        "form_id": {"type": "integer", "description": "Form id.", "minimum": 1},
    },
    required=["form_id"],
)
def get_form_questions(form_id: int) -> list[dict[str, Any]]:
    return _db_rows(
        """
        SELECT ques_id, question_captions, question_caption_hindi
        FROM public.questionl_details
        WHERE form_id = %s
        ORDER BY qus_ordering, id
        LIMIT 200
        """,
        (form_id,),
    )


@tool(
    name="list_forms",
    description="List form ids that have embedded questions, with the number of questions in each.",
    parameters={},
)
def list_forms() -> list[dict[str, Any]]:
    return _db_rows(
        """
        SELECT form_id, COUNT(DISTINCT ques_id) AS questions
        FROM public.ques_embed
        GROUP BY form_id
        ORDER BY form_id
        """
    )


@tool(
    name="query_submission_data",
    description=(
        "Answer counts, totals or filters over SUBMITTED inspection data "
        "(e.g. 'how many schools have sanitised toilets'). NOT AVAILABLE YET: "
        "the NL->SQL stage is not connected, so calling it returns an error. "
        "Call it for such questions so the user gets a clear explanation."
    ),
    parameters={
        "question": {"type": "string", "description": "The user's data question in English."},
    },
    required=["question"],
)
def query_submission_data(question: str) -> Any:
    raise NotImplementedError("the SQL/submission-data route is not connected yet")