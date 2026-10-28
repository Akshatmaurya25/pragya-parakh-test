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

LLM: OpenRouter (OPENROUTER_API_KEY / OPENROUTER_MODEL), NVIDIA
(NVIDIA_API_KEY / NVIDIA_MODEL) or local Ollama (OLLAMA_HOST / OLLAMA_MODEL);
see LLM_PROVIDER, or pick one from LLMS (the Streamlit UI does).
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
from embedder import DEFAULT_MODEL_NAME

dotenv.load_dotenv()

LOGGER = logging.getLogger("master_agent")

# LLM provider: "openrouter" (default when OPENROUTER_API_KEY is set) or "ollama".
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_URL = os.getenv("NVIDIA_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER", "openrouter" if OPENROUTER_API_KEY else "ollama"
).lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
LLM_TIMEOUT_SECONDS = 90

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


def _openai_chat(
    url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    json_mode: bool = True,
    extra: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions API and return the text."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "seed": 42,
        **(extra or {}),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS)
    if response.status_code == 400 and "response_format" in payload:
        # Some providers/free models reject JSON mode; the prompt already
        # demands JSON and _extract_json tolerates surrounding text.
        payload.pop("response_format")
        response = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError(f"{model} returned an empty response")
    return content


def openrouter_chat(system: str, user: str, json_mode: bool = True) -> str:
    """Call an OpenRouter model and return the text."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return _openai_chat(
        OPENROUTER_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, system, user, json_mode
    )


def nvidia_chat(system: str, user: str, json_mode: bool = True) -> str:
    """Call an NVIDIA-hosted model (integrate.api.nvidia.com) and return the text."""
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    return _openai_chat(
        NVIDIA_URL, NVIDIA_API_KEY, NVIDIA_MODEL, system, user, json_mode,
        # Thinking off: keeps content to the JSON decision.
        extra={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 2048},
    )


def default_llm(system: str, user: str, json_mode: bool = True) -> str:
    """Route to the configured provider."""
    if LLM_PROVIDER == "openrouter":
        return openrouter_chat(system, user, json_mode)
    if LLM_PROVIDER == "nvidia":
        return nvidia_chat(system, user, json_mode)
    return ollama_chat(system, user, json_mode)


# Display name -> provider function, for UIs that let the user choose.
LLMS: dict[str, Callable[..., str]] = {
    f"Ollama ({OLLAMA_MODEL})": ollama_chat,
    f"NVIDIA ({NVIDIA_MODEL.split('/')[-1]})": nvidia_chat,
    f"OpenRouter ({OPENROUTER_MODEL})": openrouter_chat,
}


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
- Questions about what was actually answered/submitted in a form (find \
responses, see answers) -> search_answers with form_id. If no form id is known, \
use search_questions or list_forms first, or clarify. If the user means one \
question, find its ques_id with search_questions and pass it too.
- "How was question X answered", "how many said yes" -> find the question with \
search_questions, then answer_summary(form_id, ques_id).
- To show a whole inspection/submission -> get_submission with an \
ans_submission_id taken from search_answers results.
- Other counts or totals over submitted data -> query_submission_data.
- Greetings and general questions need no tool: reply with "final".
- Never repeat an identical tool call.

How to write the "final" message (the reader is a field officer, not a developer):
- Answer in plain natural language, in the user's language. Lead with the \
conclusion or insight, then the supporting numbers. Example: "Both hostels \
that answered reported having only one of the two: either a counsellor or a \
safety workshop (score 3). None reported having both."
- Interpret the data: summarise patterns, compare counts, point out what is \
missing or unusual, and say what it means in practice.
- Do NOT show raw database content: no JSON, OptionID, ques_id, form_id, \
ans_submission_id, user_id, column names or code formatting. Refer to forms \
and questions by their name/text. Write dates in a readable form.
- Use a short markdown table only when it genuinely helps (several answer \
values with counts, a comparison, a submission's key answers). Use readable \
column headers.
- Give ids or exact raw values only if the user explicitly asks for them \
(e.g. "give me the form id", "show raw data", "exact values").
- If the data is small or inconclusive (e.g. only 2 responses), say so.
- Mention only facts that appear in tool results. Never invent answers, \
forms, questions or numbers."""


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
    def __init__(
        self,
        llm: Callable[..., str] = default_llm,
        embedding_model: str = DEFAULT_MODEL_NAME,
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
        answers: list[dict[str, Any]] = []
        seen_calls: set[str] = set()
        tool_registry.EMBEDDING_MODEL = self.embedding_model

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
                if name in ANSWER_TOOLS and outcome["ok"]:
                    answers = outcome["result"]
                continue

            message = str(decision.get("message") or "").strip()
            if action in {"final", "clarify"} and message:
                return AgentResponse(
                    "answer" if action == "final" else "clarify",
                    message,
                    last_matches if action == "final" else last_matches[:3],
                    self._trace(steps),
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