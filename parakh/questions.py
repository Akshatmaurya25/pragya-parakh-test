"""Form questions: embed them into ques_embed, search them, list them."""

import logging
from typing import Any

from parakh import db
from parakh.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
from parakh.embeddings import embed, embed_unique, to_pgvector

LOGGER = logging.getLogger(__name__)

UPSERT_QUERY = """
INSERT INTO public.ques_embed (
    form_id, department_id, ques_id, question_caption, question_caption_hindi,
    embedding_en, embedding_hi, model, created_at, updated_at
)
VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (model, ques_id) DO UPDATE
SET form_id = EXCLUDED.form_id,
    department_id = EXCLUDED.department_id,
    question_caption = EXCLUDED.question_caption,
    question_caption_hindi = EXCLUDED.question_caption_hindi,
    embedding_en = EXCLUDED.embedding_en,
    embedding_hi = EXCLUDED.embedding_hi,
    updated_at = CURRENT_TIMESTAMP
"""

# Ranks both language vectors; a question can appear twice (en + hi).
SEARCH_QUERY = """
WITH candidates AS (
    SELECT id, 'en' AS matched_language, 1 - (embedding_en <=> %(vector)s::vector) AS similarity
    FROM public.ques_embed
    WHERE model = %(model)s AND embedding_en IS NOT NULL {form_filter}
    UNION ALL
    SELECT id, 'hi' AS matched_language, 1 - (embedding_hi <=> %(vector)s::vector) AS similarity
    FROM public.ques_embed
    WHERE model = %(model)s AND embedding_hi IS NOT NULL {form_filter}
)
SELECT qe.form_id, qe.ques_id, qe.question_caption, qe.question_caption_hindi,
       c.matched_language, c.similarity
FROM candidates AS c
JOIN public.ques_embed AS qe ON qe.id = c.id
ORDER BY c.similarity DESC, qe.id, c.matched_language
LIMIT %(top_k)s
"""


def embed_questions(form_id: int | None, model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """Embed the EN + HI captions of one form (None = every form) into ques_embed."""
    form_filter = "" if form_id is None else "WHERE form_id = %s"
    questions = db.fetch(
        f"""
        SELECT form_id, department_id, ques_id, question_captions, question_caption_hindi
        FROM public.questionl_details {form_filter}
        ORDER BY form_id, qus_ordering, id
        """,
        None if form_id is None else (form_id,),
    )
    LOGGER.info("Fetched %d question(s) for form %s.", len(questions), form_id or "*")
    if not questions:
        return 0

    captions = [
        q[column]
        for q in questions
        for column in ("question_captions", "question_caption_hindi")
        if q[column]
    ]
    vectors = embed_unique(captions, model)
    LOGGER.info("Generated %d distinct embedding(s) with %s.", len(vectors), model)

    rows = [
        (
            q["form_id"], q["department_id"], q["ques_id"],
            q["question_captions"], q["question_caption_hindi"],
            vectors.get(q["question_captions"]), vectors.get(q["question_caption_hindi"]),
            EMBEDDING_MODELS[model],
        )
        for q in questions
    ]
    saved = db.execute_many(UPSERT_QUERY, rows)
    LOGGER.info("Saved %d row(s) in public.ques_embed.", saved)
    return saved


def search_questions(
    query: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    form_id: int | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Questions closest in meaning to `query`, best first."""
    params = {
        "vector": to_pgvector(embed(query.strip(), model)),
        "model": EMBEDDING_MODELS[model],
        "form_id": form_id,
        "top_k": top_k,
    }
    form_filter = "" if form_id is None else "AND form_id = %(form_id)s"
    return db.fetch(SEARCH_QUERY.format(form_filter=form_filter), params)


def get_form_questions(form_id: int) -> list[dict[str, Any]]:
    """All questions of one form, in form order."""
    return db.fetch(
        """
        SELECT ques_id, question_captions, question_caption_hindi
        FROM public.questionl_details
        WHERE form_id = %s
        ORDER BY qus_ordering, id
        LIMIT 200
        """,
        (form_id,),
    )


def list_forms(model: str = DEFAULT_EMBEDDING_MODEL) -> list[dict[str, Any]]:
    """Forms with embedded questions: form_id, form_name, questions."""
    params = (EMBEDDING_MODELS[model],)
    try:
        return db.fetch(
            """
            SELECT qe.form_id, MAX(f.form_name) AS form_name, COUNT(DISTINCT qe.ques_id) AS questions
            FROM public.ques_embed AS qe
            LEFT JOIN public.depertment_form_master AS f ON f.form_id = qe.form_id
            WHERE qe.model = %s
            GROUP BY qe.form_id
            ORDER BY qe.form_id
            """,
            params,
        )
    except Exception:  # no form master table in this DB: ids only
        return db.fetch(
            """
            SELECT form_id, NULL AS form_name, COUNT(DISTINCT ques_id) AS questions
            FROM public.ques_embed
            WHERE model = %s
            GROUP BY form_id
            ORDER BY form_id
            """,
            params,
        )
