"""Master agent: an LLM-driven tool-calling loop for Parakh.

For each user message the LLM acts as a planner. Every step it replies with
one JSON decision:

    {"action": "call_tool", "tool": "<name>", "args": {...}}
    {"action": "final",   "message": "<answer based on tool results>"}
    {"action": "clarify", "message": "<question back to the user>"}

The agent runs the tool (definitions live in tools.py), feeds the result back,
and repeats up to MAX_STEPS. This is where query enrichment (rewrite, EN/HI
variants), re-searching with a refined query, and cross-questioning happen:
the LLM sees the search results and decides what to do next.

If the LLM is unreachable or misbehaves, the agent falls back to a plain
search_questions call with a similarity-threshold check, so it is never worse
than query_resolver.py.

LLM: OpenRouter (OPENROUTER_API_KEY / OPENROUTER_MODEL) or local Ollama
(OLLAMA_HOST / OLLAMA_MODEL); see LLM_PROVIDER.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import dotenv
import requests

import tools as tool_registry

dotenv.load_dotenv()

LOGGER = logging.getLogger("master_agent")

# LLM provider: "openrouter" (default when OPENROUTER_API_KEY is set) or "ollama".
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER", "openrouter" if OPENROUTER_API_KEY else "ollama"
).lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
LLM_TIMEOUT_SECONDS = 90

MAX_STEPS = 4                # planner iterations per user message
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


# --------------------------------------------------------------------------
# LLM access
# --------------------------------------------------------------------------

def ollama_chat(system: str, user: str, json_mode: bool = True) -> str:
    """Call Ollama's chat endpoint deterministically and return the text."""
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0, "seed": 42},
    }
    if json_mode:
        payload["format"] = "json"
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat", json=payload, timeout=LLM_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def openrouter_chat(system: str, user: str, json_mode: bool = True) -> str:
    """Call an OpenRouter model (OpenAI-compatible API) and return the text."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    payload: dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "seed": 42,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    response = requests.post(
        OPENROUTER_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS
    )
    if response.status_code == 400 and "response_format" in payload:
        # Some providers/free models reject JSON mode; the prompt already
        # demands JSON and _extract_json tolerates surrounding text.
        payload.pop("response_format")
        response = requests.post(
            OPENROUTER_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS
        )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError("OpenRouter returned an empty response")
    return content


def default_llm(system: str, user: str, json_mode: bool = True) -> str:
    """Route to the configured provider."""
    if LLM_PROVIDER == "openrouter":
        return openrouter_chat(system, user, json_mode)
    return ollama_chat(system, user, json_mode)


def _extract_json(raw: str) -> Any:
    """Parse JSON, tolerating code fences or reasoning text around the object."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


# --------------------------------------------------------------------------
# Planner prompt
# --------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the master agent of Parakh, a Madhya Pradesh \
government inspection-form system. Forms cover schools, anganwadis, hospitals, \
hostels and aadhar centres; form fields are questions such as toilet \
availability, teacher presence, mid-day meal menu, drinking water. Users write \
English, Hindi or Hinglish.

You cannot run code or read files yourself. You act ONLY through these tools:

{tools}

Reply with ONLY one JSON object, one of:
  {{"action": "call_tool", "tool": "<tool name>", "args": {{...}}}}
  {{"action": "final", "message": "<reply to the user>"}}
  {{"action": "clarify", "message": "<one short question to the user>"}}

Rules:
- Any request to find, search or retrieve form fields/questions is a \
search_questions call. Rewrite the user's message into one clear English \
sentence, resolve follow-ups using the conversation, and add 2-4 variants \
(English and Hindi). If the user asks for a number of results ("top 4"), pass \
it as top_k; if they name a form, pass form_id.
- Read the tool results. If the matches do not really cover the user's topic, \
call search_questions once more with a better query. If it is still unclear, \
use "clarify" and offer the closest matches as options.
- If the message is too vague to search (a single word like "status", or \
"retrieve the top 4" with no topic), use "clarify".
- Counts or totals over submitted data -> query_submission_data.
- Greetings and general questions need no tool: reply with "final".
- In "final", mention only fields that appear in tool results. Answer in the \
user's language. Never invent form ids or question text.
- Never repeat an identical tool call."""


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


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

def _heuristic_is_good(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


class MasterAgent:
    def __init__(self, llm: Callable[..., str] = default_llm) -> None:
        self.llm = llm
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
        system = PLANNER_SYSTEM.format(tools=tool_registry.describe_tools())
        user = (
            f"Conversation so far:\n{_format_history(self.history)}\n\n"
            f"User message: {query}\n\n"
            f"Tool calls made for this message:\n{_format_steps(steps)}\n\n"
            "Next decision (JSON only):"
        )
        data = _extract_json(self.llm(system, user, json_mode=True))
        if not isinstance(data, dict):
            raise ValueError("planner did not return a JSON object")
        return data

    def _run(self, query: str) -> AgentResponse:
        steps: list[dict[str, Any]] = []
        last_matches: list[dict[str, Any]] = []
        seen_calls: set[str] = set()

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
                outcome = tool_registry.call_tool(str(name), args)
                steps.append({"tool": name, "args": args, "outcome": outcome})
                if name == "search_questions" and outcome["ok"]:
                    last_matches = outcome["result"]
                continue

            message = str(decision.get("message") or "").strip()
            if action in {"final", "clarify"} and message:
                return AgentResponse(
                    "answer" if action == "final" else "clarify",
                    message,
                    last_matches if action == "final" else last_matches[:3],
                    self._trace(steps),
                )
            LOGGER.warning("planner returned an unusable decision: %r", decision)
            break

        return self._fallback(query, steps, last_matches)

    def _fallback(
        self,
        query: str,
        steps: list[dict[str, Any]],
        matches: list[dict[str, Any]],
    ) -> AgentResponse:
        """Plain single-query search when the planner cannot finish."""
        if not matches:
            outcome = tool_registry.call_tool("search_questions", {"query": query})
            steps.append({"tool": "search_questions", "args": {"query": query}, "outcome": outcome})
            matches = outcome["result"] if outcome["ok"] else []
        good = _heuristic_is_good(matches)
        trace = self._trace(steps)
        trace["fallback"] = True
        if good:
            return AgentResponse("answer", "Matching form fields:\n" + _describe_matches(good), good, trace)
        message = "I could not find a clear match. Could you tell me more about what you want to find?"
        if matches:
            message += " Closest fields I found:\n" + _describe_matches(matches[:3])
        return AgentResponse("clarify", message, matches[:3], trace)

    @staticmethod
    def _trace(steps: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "steps": [
                {"tool": s["tool"], "args": s["args"], "ok": s["outcome"]["ok"]}
                for s in steps
            ]
        }


def main() -> None:
    """Interactive loop. Clarifying questions are answered in the next turn."""
    agent = MasterAgent()
    print("Parakh master agent. Ctrl+C to quit.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue
        result = agent.handle(text)
        print(f"\n[{result.action}] {result.message}")
        print(json.dumps(result.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()