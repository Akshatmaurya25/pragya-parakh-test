"""Master agent: an LLM-driven tool-calling loop.

For each user message the LLM acts as a planner. Every step it replies with
one JSON decision:

    {"action": "call_tool", "tool": "<name>", "args": {...}}
    {"action": "final",   "message": "<answer based on tool results>"}
    {"action": "clarify", "message": "<question back to the user>"}

The agent runs the tool, feeds the result back, and repeats up to MAX_STEPS.
If the LLM is unreachable or misbehaves, it falls back to a plain
search_questions call with a similarity-threshold check.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from parakh.agent import tools
from parakh.agent.llm import default_llm, extract_json
from parakh.agent.prompts import PLANNER_SYSTEM
from parakh.agent.registry import call_tool, describe_tools
from parakh.config import DEFAULT_EMBEDDING_MODEL

LOGGER = logging.getLogger(__name__)

MAX_STEPS = 6                # planner iterations per user message (answer flows chain 3-4 tools)
ANSWER_TOOLS = {"search_answers", "get_submission", "answer_summary"}
MAX_RESULT_CHARS = 4000      # tool output shown back to the LLM
# Fallback (no-LLM) thresholds. Starting values: calibrate on your own
# labelled queries, similarity scales differ per embedding model.
MIN_SIMILARITY = 0.55
MIN_MARGIN = 0.02


@dataclass
class AgentResponse:
    action: str                      # answer | clarify
    message: str
    matches: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    answers: list[dict[str, Any]] = field(default_factory=list)  # rows from the last answer tool


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])


def _format_steps(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "(none yet)"
    parts = []
    for i, step in enumerate(steps, 1):
        outcome = json.dumps(step["outcome"], ensure_ascii=False, default=str)
        if len(outcome) > MAX_RESULT_CHARS:
            outcome = outcome[:MAX_RESULT_CHARS] + "...(truncated)"
        parts.append(
            f"{i}. call {step['tool']} {json.dumps(step['args'], ensure_ascii=False)}\n"
            f"   result: {outcome}"
        )
    return "\n".join(parts)


def _strong_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No-LLM check: keep top hits only if clearly strong and separated."""
    if not matches:
        return []
    top = matches[0]["best_similarity"]
    second = matches[1]["best_similarity"] if len(matches) > 1 else 0.0
    if top >= MIN_SIMILARITY and (top - second >= MIN_MARGIN or top >= 0.75):
        return [m for m in matches if m["best_similarity"] >= max(MIN_SIMILARITY, top - 0.1)]
    return []


def _describe_matches(matches: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- [form {m.get('form_id')}, q {m.get('ques_id')}] "
        f"{m.get('question_caption') or m.get('question_caption_hindi')}"
        for m in matches
    )


def _trace(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"steps": [{"tool": s["tool"], "args": s["args"], "ok": s["outcome"]["ok"]} for s in steps]}


class MasterAgent:
    def __init__(
        self,
        llm: Callable[..., str] = default_llm,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
        self.history: list[dict[str, str]] = []

    def handle(self, query: str) -> AgentResponse:
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        response = self._run(query)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response.message})
        return response

    def _decide(self, query: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        system = PLANNER_SYSTEM.format(tools=describe_tools())
        user = (
            f"Conversation so far:\n{_format_history(self.history)}\n\n"
            f"User message: {query}\n\n"
            f"Tool calls made for this message:\n{_format_steps(steps)}\n\n"
            "Next decision (JSON only):"
        )
        data = extract_json(self.llm(system, user, json_mode=True))
        if not isinstance(data, dict):
            raise ValueError("planner did not return a JSON object")
        return data

    def _run(self, query: str) -> AgentResponse:
        steps: list[dict[str, Any]] = []
        last_matches: list[dict[str, Any]] = []
        answers: list[dict[str, Any]] = []
        seen_calls: set[str] = set()
        tools.EMBEDDING_MODEL = self.embedding_model

        for _ in range(MAX_STEPS):
            try:
                decision = self._decide(query, steps)
            except Exception as error:  # LLM down, bad JSON, timeout
                LOGGER.warning("planner failed, using fallback: %s", error)
                break

            action = decision.get("action")
            if action == "call_tool":
                name = decision.get("tool")
                args = decision.get("args") or {}
                signature = json.dumps([name, args], sort_keys=True, default=str)
                if signature in seen_calls:
                    LOGGER.warning("planner repeated a tool call, using fallback")
                    break
                seen_calls.add(signature)
                outcome = call_tool(str(name), args)
                steps.append({"tool": name, "args": args, "outcome": outcome})
                if name == "search_questions" and outcome["ok"]:
                    last_matches = outcome["result"]
                if name in ANSWER_TOOLS and outcome["ok"]:
                    answers = outcome["result"]
                continue

            message = str(decision.get("message") or "").strip()
            if action in {"final", "clarify"} and message:
                return AgentResponse(
                    "answer" if action == "final" else "clarify",
                    message,
                    last_matches if action == "final" else last_matches[:3],
                    _trace(steps),
                    answers,
                )
            LOGGER.warning("planner returned an unusable decision: %r", decision)
            break

        response = self._fallback(query, steps, last_matches)
        response.answers = answers
        return response

    def _fallback(
        self,
        query: str,
        steps: list[dict[str, Any]],
        matches: list[dict[str, Any]],
    ) -> AgentResponse:
        """Plain single-query search when the planner cannot finish."""
        if not matches:
            outcome = call_tool("search_questions", {"query": query})
            steps.append({"tool": "search_questions", "args": {"query": query}, "outcome": outcome})
            matches = outcome["result"] if outcome["ok"] else []
        good = _strong_matches(matches)
        trace = _trace(steps)
        trace["fallback"] = True
        if good:
            return AgentResponse("answer", "Matching form fields:\n" + _describe_matches(good), good, trace)
        message = "I could not find a clear match. Could you tell me more about what you want to find?"
        if matches:
            message += " Closest fields I found:\n" + _describe_matches(matches[:3])
        return AgentResponse("clarify", message, matches[:3], trace)
