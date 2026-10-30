"""Every tool the agent can call. Thin wrappers over parakh.questions / parakh.answers.

To add a tool, write a function here and decorate it with @tool (see registry.py).
"""

import logging
from typing import Any

from parakh import answers, questions
from parakh.agent.registry import tool
from parakh.config import DEFAULT_EMBEDDING_MODEL

LOGGER = logging.getLogger(__name__)

RRF_K = 60  # reciprocal-rank-fusion constant

# Embedding model used by search tools; MasterAgent sets it per run.
# ponytail: module global, not safe for concurrent multi-user sessions; thread
# it through call_tool if this is ever served to several users at once.
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL


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
            rows = questions.search_questions(
                phrasing, EMBEDDING_MODEL, form_id=form_id, top_k=max(top_k * 2, 10)
            )
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
    return questions.get_form_questions(form_id)


@tool(
    name="list_forms",
    description="List forms that have embedded questions: form id, form name and number of questions.",
    parameters={},
)
def list_forms() -> list[dict[str, Any]]:
    return questions.list_forms(EMBEDDING_MODEL)


@tool(
    name="search_answers",
    description=(
        "Semantic search over SUBMITTED answers of one form. Use when the user "
        "asks what was answered or wants responses mentioning something. Needs "
        "a form_id; get it from list_forms or search_questions first. Pass "
        "ques_id to search the answers of one question only."
    ),
    parameters={
        "query": {"type": "string", "description": "What the answer should say (English or Hindi)."},
        "form_id": {"type": "integer", "description": "Form id.", "minimum": 1},
        "top_k": {
            "type": "integer",
            "description": "How many answers to return (default 10).",
            "minimum": 1,
            "maximum": 50,
        },
        "ques_id": {"type": "integer", "description": "Restrict to one question id.", "minimum": 1},
    },
    required=["query", "form_id"],
)
def search_answers(
    query: str, form_id: int, top_k: int = 10, ques_id: int | None = None
) -> list[dict[str, Any]]:
    return answers.search_answers(query, form_id, EMBEDDING_MODEL, top_k=top_k, ques_id=ques_id)


@tool(
    name="get_submission",
    description=(
        "All answers of one submitted inspection, with question text, in form order. "
        "Use after search_answers when the user wants the full submission/report."
    ),
    parameters={
        "ans_submission_id": {"type": "string", "description": "Submission id from search_answers."},
    },
    required=["ans_submission_id"],
)
def get_submission(ans_submission_id: str) -> list[dict[str, Any]]:
    return answers.get_submission(ans_submission_id)


@tool(
    name="answer_summary",
    description=(
        "How one question was answered across all submissions: each distinct "
        "answer value with its count, most common first. Use for 'how many said "
        "yes', 'what are the common answers to X'. Get ques_id from search_questions."
    ),
    parameters={
        "form_id": {"type": "integer", "description": "Form id.", "minimum": 1},
        "ques_id": {"type": "integer", "description": "Question id.", "minimum": 1},
    },
    required=["form_id", "ques_id"],
)
def answer_summary(form_id: int, ques_id: int) -> list[dict[str, Any]]:
    return answers.answer_summary(form_id, ques_id)


@tool(
    name="query_submission_data",
    description=(
        "Complex counts, totals or filters over SUBMITTED inspection data that "
        "answer_summary cannot cover (several questions, districts, dates). NOT AVAILABLE YET: "
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
