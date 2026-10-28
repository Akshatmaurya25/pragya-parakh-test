"""Embed submitted answers (ans_value) of one form into ``ans_embed`` and search them."""

import json
import logging
import os
import re
import sys
from typing import Any, Callable
from urllib.parse import quote

from db_conn import execute, execute_many
from embedder import DEFAULT_MODEL_NAME, MODEL_NAME, MODEL_STORAGE_NAMES, embedder
from form_embedder import LOGGER, _to_pgvector_literal, configure_logging, log_message

ANSWER_QUERY = """
SELECT id, ans_submissiom_id, form_id, department_id, ques_id, user_id,
       ans_value, insert_dt
FROM public.form_respond_details
WHERE form_id = %s AND NULLIF(TRIM(ans_value), '') IS NOT NULL
ORDER BY id
"""

INSERT_QUERY = """
INSERT INTO public.ans_embed (
    response_id, ans_submission_id, form_id, department_id, ques_id, user_id,
    ans_value, embedding, model, insert_dt, created_at, updated_at
)
VALUES (
    %s, %s, %s, %s, %s, %s,
    %s, %s::vector, %s, %s,
    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
)
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
       1 - (ae.embedding <=> %s::vector) AS similarity
FROM public.ans_embed AS ae
LEFT JOIN public.ques_embed AS qe ON qe.model = ae.model AND qe.ques_id = ae.ques_id
WHERE ae.model = %s AND ae.form_id = %s {ques_filter}
ORDER BY ae.embedding <=> %s::vector
LIMIT %s
"""


def _check_form_id(form_id: Any) -> None:
    if not isinstance(form_id, int) or isinstance(form_id, bool) or form_id < 1:
        raise ValueError("form_id must be a positive integer")


def answer_text(raw: Any) -> str:
    """Readable answer text; option answers are JSON like [{"OptionID":"1","OptionText":"Rural"}]."""
    text = str(raw or "").strip()
    if text.startswith("["):
        try:
            options = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(options, list) and all(isinstance(o, dict) for o in options):
            return ", ".join(
                str(o.get("OptionText") or "").strip()
                for o in options
                if str(o.get("OptionText") or "").strip()
            )
    return text


# Photos are stored as relative paths ("Citizen/SurveyImages/x.jpg"); this
# server prefix turns them into viewable URLs. Unset -> no photos shown.
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
IMAGE_PATH = re.compile(r'[^"\\\[\]{},]+?\.(?:jpe?g|png|webp|gif)\b', re.IGNORECASE)


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


def embed_distinct(
    texts: list[str], embed: Callable[[list[str]], Any]
) -> dict[str, list[float]]:
    """Embed each distinct text once; answers like 'Yes'/'हाँ' repeat a lot."""
    unique = list(dict.fromkeys(texts))
    if not unique:
        return {}
    vectors = embed(unique)
    if len(vectors) != len(unique):
        raise ValueError(f"Expected {len(unique)} embeddings")
    return {text: [float(v) for v in vector] for text, vector in zip(unique, vectors)}


def embed_answers(form_id: int, model_name: str = MODEL_NAME) -> int:
    """Fetch, embed and upsert every non-empty answer of one form."""
    _check_form_id(form_id)
    configure_logging()
    log_message(f"Embedding answers for form_id={form_id} with {model_name}.")

    rows = json.loads(execute(ANSWER_QUERY, (form_id,)))
    log_message(f"Fetched {len(rows)} answer(s).")
    pairs = [(row, answer_text(row["ans_value"])) for row in rows]
    pairs = [(row, value) for row, value in pairs if value]  # drop empty option answers
    if not pairs:
        return 0
    rows = [row for row, _ in pairs]
    values = [value for _, value in pairs]
    vectors = embed_distinct(values, lambda texts: embedder(texts, model_name=model_name))
    log_message(f"Generated {len(vectors)} distinct embedding(s).")

    storage_model = MODEL_STORAGE_NAMES[model_name]
    insert_rows = [
        (
            row["id"], row.get("ans_submissiom_id"), row["form_id"],
            row.get("department_id"), row.get("ques_id"), row.get("user_id"),
            value, _to_pgvector_literal(vectors[value]), storage_model,
            row.get("insert_dt"),
        )
        for row, value in zip(rows, values)
    ]
    saved = execute_many(INSERT_QUERY, insert_rows)
    log_message(f"Saved {saved} row(s) in public.ans_embed.")
    return saved


def retrieve_answers(
    query_text: str,
    form_id: int,
    top_k: int = 10,
    ques_id: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Return the answers of one form closest in meaning to ``query_text``."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    _check_form_id(form_id)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if model_name not in MODEL_STORAGE_NAMES:
        raise ValueError(f"Unsupported embedding model: {model_name}")

    vector = _to_pgvector_literal(
        embedder(query_text.strip(), model_name=model_name).tolist()
    )
    params: list[Any] = [vector, MODEL_STORAGE_NAMES[model_name], form_id]
    if ques_id is not None:
        params.append(ques_id)
    params += [vector, top_k]
    query = SEARCH_QUERY.format(ques_filter="AND ae.ques_id = %s" if ques_id is not None else "")
    return json.loads(execute(query, tuple(params)))


def _selftest() -> None:
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(len(t))] for t in texts]

    result = embed_distinct(["Yes", "No", "Yes", "हाँ"], fake_embed)
    assert calls == [["Yes", "No", "हाँ"]], calls
    assert result == {"Yes": [3.0], "No": [2.0], "हाँ": [3.0]}, result
    assert embed_distinct([], fake_embed) == {}
    assert answer_text('[{"OptionID":"1","OptionText":"Rural"}]') == "Rural"
    assert answer_text('[{"OptionID":"1","OptionText":"A"},{"OptionID":"2","OptionText":"B"}]') == "A, B"
    assert answer_text('[{"OptionID":"","OptionText":""}]') == ""
    assert answer_text(" free text ") == "free text"
    assert answer_text("[not json") == "[not json"
    base = "https://img.example"
    plain = '[{"OptionID":"jpg,77.42,23.24","OptionText":"Citizen/SurveyImages/9_1184_SurveyImg.jpg"}]'
    assert image_urls(plain, base) == [f"{base}/Citizen/SurveyImages/9_1184_SurveyImg.jpg"]
    nested = (
        r'[{"OptionID":"0","OptionText":"[[{\"answerValue\":{\"optionID\":\"png,79.89,22.02\",'
        r'\"optionText\":\"Tribal Affairs Department/SurveyImages/0_5338.png\"}}]]"}]'
    )
    assert image_urls(nested, base) == [f"{base}/Tribal%20Affairs%20Department/SurveyImages/0_5338.png"]
    assert image_urls(plain, "") == []
    assert image_urls('[{"OptionID":"1","OptionText":"Rural"}]', base) == []
    print("selftest ok")


def main() -> None:
    """Prompt for a form ID and embed its answers."""
    configure_logging()
    try:
        embed_answers(int(input("Enter form ID: ").strip()))
    except Exception as error:
        log_message(f"Answer embedding failed: {error}", logging.ERROR)
        LOGGER.exception("Answer embedding traceback")
        raise


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
