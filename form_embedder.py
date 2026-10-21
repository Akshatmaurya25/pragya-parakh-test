"""Create and store bilingual question embeddings for one form."""

import json
import logging
from pathlib import Path
from typing import Any

from db_conn import execute, execute_many
from embedder import MODEL_NAME, embedder
import dotenv

dotenv.load_dotenv()  

LOGGER = logging.getLogger("form_embedder")
LOG_FILE = Path(__file__).with_name("form_embedder.log")

QUESTION_QUERY = """
SELECT form_id, department_id, ques_id,
       question_captions, question_caption_hindi
FROM public.questionl_details
WHERE form_id = %s
ORDER BY qus_ordering, id
"""

ALL_QUESTIONS_QUERY = """
SELECT form_id, department_id, ques_id,
       question_captions, question_caption_hindi
FROM public.questionl_details
ORDER BY form_id, qus_ordering, id
"""

INSERT_QUERY = """
INSERT INTO public.ques_embed (
    form_id,
    department_id,
    ques_id,
    question_caption,
    question_caption_hindi,
    embedding_en,
    embedding_hi,
    model,
    created_at,
    updated_at
)
VALUES (
    %s, %s, %s, %s, %s,
    %s::vector, %s::vector, %s,
    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
)
ON CONFLICT (model, ques_id) DO UPDATE
SET form_id = EXCLUDED.form_id,
    department_id = EXCLUDED.department_id,
    question_caption = EXCLUDED.question_caption,
    question_caption_hindi = EXCLUDED.question_caption_hindi,
    embedding_en = EXCLUDED.embedding_en,
    embedding_hi = EXCLUDED.embedding_hi,
    updated_at = CURRENT_TIMESTAMP
"""


def configure_logging() -> None:
    """Write module activity to a persistent log file without duplicate handlers."""
    if any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == LOG_FILE.resolve()
        for handler in LOGGER.handlers
    ):
        return

    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    LOGGER.addHandler(handler)


def log_message(message: str, level: int = logging.INFO) -> None:
    """Print a progress message and write the same message to the log file."""
    print(message)
    LOGGER.log(level, message)


def fetch_form_questions(form_id: int | None) -> list[dict[str, Any]]:
    """Fetch ordered captions for one form, or every question if ID is None."""
    if form_id is None:
        log_message("Fetching questions for all forms.")
        rows = execute(ALL_QUESTIONS_QUERY)
    else:
        log_message(f"Fetching questions for form_id={form_id}.")
        rows = execute(QUESTION_QUERY, (form_id,))

    decoded_rows = json.loads(rows)
    if not isinstance(decoded_rows, list) or not all(
        isinstance(row, dict) for row in decoded_rows
    ):
        raise TypeError("Database query must return a JSON list of row objects")

    log_message(f"Fetched {len(decoded_rows)} question(s).")
    return decoded_rows


def _embed_caption_column(
    rows: list[dict[str, Any]], column: str
) -> dict[int, list[float]]:
    """Embed non-empty captions and map vectors back to their question rows."""
    row_indices: list[int] = []
    captions: list[str] = []

    for index, row in enumerate(rows):
        caption = row.get(column)
        if caption is None or caption == "":
            continue
        if not isinstance(caption, str):
            raise TypeError(f"{column} must contain only strings or None")
        row_indices.append(index)
        captions.append(caption)

    if not captions:
        log_message(f"No captions found for {column}; storing NULL embeddings.")
        return {}

    log_message(f"Generating {len(captions)} embedding(s) for {column}.")
    vectors = embedder(captions)
    if len(vectors) != len(row_indices):
        raise ValueError(f"Expected {len(row_indices)} embeddings for {column}")

    return {
        index: vector.tolist()
        for index, vector in zip(row_indices, vectors)
    }


def _to_pgvector_literal(vector: list[float] | None) -> str | None:
    """Convert an embedding to PostgreSQL's vector input format."""
    if vector is None:
        return None
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def prepare_embedding_rows(
    form_id: int | None, questions: list[dict[str, Any]]
) -> list[tuple[Any, ...]]:
    """Build target-table rows, embedding each available caption language."""
    english_embeddings = _embed_caption_column(questions, "question_captions")
    hindi_embeddings = _embed_caption_column(questions, "question_caption_hindi")
    insert_rows: list[tuple[Any, ...]] = []

    for index, question in enumerate(questions):
        ques_id = question.get("ques_id")
        if ques_id is None:
            raise ValueError("Every question row must have a ques_id")

        insert_rows.append(
            (
                question.get("form_id", form_id),
                question.get("department_id"),
                ques_id,
                question.get("question_captions"),
                question.get("question_caption_hindi"),
                _to_pgvector_literal(english_embeddings.get(index)),
                _to_pgvector_literal(hindi_embeddings.get(index)),
                MODEL_NAME,
            )
        )

    return insert_rows


def insert_form_embeddings(insert_rows: list[tuple[Any, ...]]) -> int:
    """Insert prepared question captions and vectors into ``ques_embed``."""
    if not insert_rows:
        log_message("There are no question rows to insert.")
        return 0

    log_message(
        f"Inserting or updating {len(insert_rows)} row(s) in public.ques_embed."
    )
    affected_count = execute_many(INSERT_QUERY, insert_rows)
    log_message(f"Saved {affected_count} row(s) in public.ques_embed.")
    return affected_count


def embed_form(form_id: int | None) -> int:
    """Fetch, embed, and insert a form's questions or the entire question table."""
    if form_id is not None and (
        not isinstance(form_id, int) or isinstance(form_id, bool) or form_id < 1
    ):
        raise ValueError("form_id must be a positive integer or None")

    configure_logging()
    log_message(f"Embedding model: {MODEL_NAME}.")
    if form_id is None:
        log_message("Starting embedding job for all forms.")
    else:
        log_message(f"Starting embedding job for form_id={form_id}.")
    questions = fetch_form_questions(form_id)
    insert_rows = prepare_embedding_rows(form_id, questions)
    inserted_count = insert_form_embeddings(insert_rows)
    if form_id is None:
        log_message("Embedding job completed for all forms.")
    else:
        log_message(f"Embedding job completed for form_id={form_id}.")
    return inserted_count


def _parse_form_id(value: str) -> int | None:
    """Interpret an asterisk as a request to process all forms."""
    if value.strip() == "*":
        return None
    try:
        form_id = int(value)
    except ValueError as error:
        raise ValueError("Enter a positive form ID or * for all forms") from error
    if form_id < 1:
        raise ValueError("Enter a positive form ID or * for all forms")
    return form_id


def main() -> None:
    """Prompt for a form ID or * and run the corresponding embedding job."""
    configure_logging()
    try:
        form_id_text = input("Enter form ID (* for all forms): ").strip()
        form_id = _parse_form_id(form_id_text)
        embed_form(form_id)
    except (TypeError, ValueError) as error:
        log_message(f"Invalid form ID or question data: {error}", logging.ERROR)
        raise
    except Exception as error:
        log_message(f"Embedding job failed: {error}", logging.ERROR)
        LOGGER.exception("Embedding job traceback")
        raise


if __name__ == "__main__":
    main()
