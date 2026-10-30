"""Submitted answers: make them readable, embed them into ans_embed, search and summarise them."""

import json
import logging
import re
from typing import Any
from urllib.parse import quote

from parakh import db
from parakh.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, IMAGE_BASE_URL
from parakh.embeddings import embed, embed_unique, to_pgvector

LOGGER = logging.getLogger(__name__)

IMAGE_PATH = re.compile(r'[^"\\\[\]{},]+?\.(?:jpe?g|png|webp|gif)\b', re.IGNORECASE)

UPSERT_QUERY = """
INSERT INTO public.ans_embed (
    response_id, ans_submission_id, form_id, department_id, ques_id, user_id,
    ans_value, embedding, model, insert_dt, created_at, updated_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (model, response_id) DO UPDATE
SET ans_submission_id = EXCLUDED.ans_submission_id,
    form_id = EXCLUDED.form_id,
    department_id = EXCLUDED.department_id,
    ques_id = EXCLUDED.ques_id,
    user_id = EXCLUDED.user_id,
    ans_value = EXCLUDED.ans_value,
    embedding = EXCLUDED.embedding,
    insert_dt = EXCLUDED.insert_dt,
    updated_at = CURRENT_TIMESTAMP
"""

SEARCH_QUERY = """
SELECT ae.response_id, ae.ans_submission_id, ae.form_id, ae.ques_id, ae.user_id,
       ae.ans_value, ae.insert_dt,
       qe.question_caption, qe.question_caption_hindi,
       1 - (ae.embedding <=> %(vector)s::vector) AS similarity
FROM public.ans_embed AS ae
LEFT JOIN public.ques_embed AS qe ON qe.model = ae.model AND qe.ques_id = ae.ques_id
WHERE ae.model = %(model)s AND ae.form_id = %(form_id)s {ques_filter}
ORDER BY ae.embedding <=> %(vector)s::vector
LIMIT %(top_k)s
"""


# --------------------------------------------------------------------------
# Readable answers
# --------------------------------------------------------------------------

def answer_text(raw: Any) -> str:
    """Readable answer; option answers are JSON like [{"OptionID":"1","OptionText":"Rural"}]."""
    text = str(raw or "").strip()
    if text.startswith("["):
        try:
            options = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(options, list) and all(isinstance(o, dict) for o in options):
            texts = (str(o.get("OptionText") or "").strip() for o in options)
            return ", ".join(t for t in texts if t)
    return text


def image_urls(raw: Any, base_url: str | None = None) -> list[str]:
    """Photo URLs in an answer: a plain image option or nested in add-more JSON."""
    base = (IMAGE_BASE_URL if base_url is None else base_url).rstrip("/")
    urls = []
    for path in dict.fromkeys(p.strip() for p in IMAGE_PATH.findall(str(raw or ""))):
        if path.lower().startswith(("http://", "https://")):
            urls.append(path)
        elif base:
            urls.append(f"{base}/{quote(path.lstrip('/'))}")
    return urls


def _readable(row: dict[str, Any]) -> dict[str, Any]:
    """Photo answers -> row["photos"] URLs; option JSON -> plain text."""
    photos = image_urls(row.get("ans_value"))
    if photos:
        row["photos"] = photos
        row["ans_value"] = "photo"
    else:
        row["ans_value"] = answer_text(row.get("ans_value"))
    return row


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def answered_form_ids() -> list[int]:
    rows = db.fetch("SELECT DISTINCT form_id FROM public.form_respond_details ORDER BY form_id")
    return [row["form_id"] for row in rows]


def embed_answers(form_id: int, model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """Embed every non-empty answer of one form into ans_embed."""
    rows = db.fetch(
        """
        SELECT id, ans_submissiom_id, form_id, department_id, ques_id, user_id, ans_value, insert_dt
        FROM public.form_respond_details
        WHERE form_id = %s AND NULLIF(TRIM(ans_value), '') IS NOT NULL
        ORDER BY id
        """,
        (form_id,),
    )
    LOGGER.info("Fetched %d answer(s) for form %s.", len(rows), form_id)
    pairs = [(row, answer_text(row["ans_value"])) for row in rows]
    pairs = [(row, text) for row, text in pairs if text]  # drop empty option answers
    if not pairs:
        return 0

    vectors = embed_unique([text for _, text in pairs], model)
    LOGGER.info("Generated %d distinct embedding(s) with %s.", len(vectors), model)
    insert_rows = [
        (
            row["id"], row["ans_submissiom_id"], row["form_id"], row["department_id"],
            row["ques_id"], row["user_id"], text, vectors[text], EMBEDDING_MODELS[model],
            row["insert_dt"],
        )
        for row, text in pairs
    ]
    saved = db.execute_many(UPSERT_QUERY, insert_rows)
    LOGGER.info("Saved %d row(s) in public.ans_embed.", saved)
    return saved


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def search_answers(
    query: str,
    form_id: int,
    model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 10,
    ques_id: int | None = None,
) -> list[dict[str, Any]]:
    """Answers of one form closest in meaning to `query`, best first."""
    params = {
        "vector": to_pgvector(embed(query.strip(), model)),
        "model": EMBEDDING_MODELS[model],
        "form_id": form_id,
        "ques_id": ques_id,
        "top_k": top_k,
    }
    ques_filter = "" if ques_id is None else "AND ae.ques_id = %(ques_id)s"
    rows = db.fetch(SEARCH_QUERY.format(ques_filter=ques_filter), params)
    for row in rows:
        _readable(row)
        row["similarity"] = round(float(row["similarity"]), 3)
    return rows


def get_submission(ans_submission_id: str) -> list[dict[str, Any]]:
    """All answers of one submitted inspection, with question text, in form order."""
    rows = db.fetch(
        """
        SELECT fr.form_id, fr.ques_id, q.question_captions, q.question_caption_hindi,
               fr.ans_value, fr.user_id, fr.insert_dt
        FROM public.form_respond_details AS fr
        LEFT JOIN public.questionl_details AS q
               ON q.form_id = fr.form_id AND q.ques_id = fr.ques_id
        WHERE fr.ans_submissiom_id = %s
        ORDER BY q.qus_ordering, fr.id
        LIMIT 300
        """,
        (ans_submission_id,),
    )
    return [_readable(row) for row in rows]


def answer_summary(form_id: int, ques_id: int) -> list[dict[str, Any]]:
    """Each distinct answer to one question with its count, most common first."""
    rows = db.fetch(
        """
        SELECT TRIM(ans_value) AS ans_value, COUNT(*) AS responses
        FROM public.form_respond_details
        WHERE form_id = %s AND ques_id = %s AND NULLIF(TRIM(ans_value), '') IS NOT NULL
        GROUP BY TRIM(ans_value)
        ORDER BY responses DESC
        LIMIT 500
        """,
        (form_id, ques_id),
    )
    # Option answers are JSON; several raw values can share one readable text.
    counts: dict[str, int] = {}
    for row in rows:
        text = answer_text(row["ans_value"]) or "(blank)"
        counts[text] = counts.get(text, 0) + int(row["responses"])
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:50]
    return [{"answer": text, "responses": n} for text, n in ranked]
