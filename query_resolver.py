"""Search stored question embeddings and print matching question rows as JSON."""

import json
from typing import Any

from db_conn import execute
from embedder import (
    DEFAULT_MODEL_NAME,
    MODEL_OPTIONS,
    MODEL_STORAGE_NAMES,
    embedder,
)


def _to_pgvector_literal(vector: list[float]) -> str:
    """Convert an embedding to PostgreSQL's vector input format."""
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _build_search_query(form_id: int | None) -> str:
    """Build a model-filtered query ranking both language embeddings."""
    form_filter = "AND form_id = %s" if form_id is not None else ""
    return f"""
    WITH candidates AS (
        SELECT id, 'en' AS matched_language,
               1 - (embedding_en <=> %s::vector) AS similarity
        FROM public.ques_embed
        WHERE model = %s AND embedding_en IS NOT NULL {form_filter}
        UNION ALL
        SELECT id, 'hi' AS matched_language,
               1 - (embedding_hi <=> %s::vector) AS similarity
        FROM public.ques_embed
        WHERE model = %s AND embedding_hi IS NOT NULL {form_filter}
    )
    SELECT qe.*, candidates.matched_language, candidates.similarity
    FROM candidates
    JOIN public.ques_embed AS qe ON qe.id = candidates.id
    ORDER BY candidates.similarity DESC, qe.id, candidates.matched_language
    LIMIT %s
    """


def retrieve_top_k(
    query_text: str,
    form_id: int | None = None,
    top_k: int = 3,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Return the closest stored question-embedding rows for a text query.

    Both English and Hindi embeddings are searched with the same multilingual
    query vector. When ``form_id`` is omitted, candidates from every form are
    included. Results represent the nearest vectors, so a row with both
    embeddings can appear more than once if both vectors rank in the top k.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    if form_id is not None and (
        not isinstance(form_id, int) or isinstance(form_id, bool) or form_id < 1
    ):
        raise ValueError("form_id must be a positive integer or None")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    supported_models = {model_id for _, model_id in MODEL_OPTIONS.values()}
    if model_name not in supported_models:
        raise ValueError(f"Unsupported embedding model: {model_name}")
    storage_model_name = MODEL_STORAGE_NAMES[model_name]

    query_vector = _to_pgvector_literal(
        embedder(query_text.strip(), model_name=model_name).tolist()
    )
    query = _build_search_query(form_id)

    parameters: tuple[Any, ...]
    if form_id is None:
        parameters = (
            query_vector,
            storage_model_name,
            query_vector,
            storage_model_name,
            top_k,
        )
    else:
        parameters = (
            query_vector,
            storage_model_name,
            form_id,
            query_vector,
            storage_model_name,
            form_id,
            top_k,
        )

    rows = json.loads(execute(query, parameters))
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise TypeError("Database query must return a JSON list of row objects")
    return rows


def resolve_query(
    query_text: str,
    form_id: int | None = None,
    top_k: int = 3,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Search, print compact matches, and return the complete rows."""
    results = retrieve_top_k(
        query_text,
        form_id=form_id,
        top_k=top_k,
        model_name=model_name,
    )
    display_results = []
    for row in results:
        display_row = {
            key: value
            for key, value in row.items()
            if key not in {"embedding_en", "embedding_hi", "similarity"}
        }
        similarity = row.get("similarity")
        if similarity is not None:
            display_row["similarity_percentage"] = (
                f"{float(similarity) * 100:.2f}%"
            )
        display_results.append(display_row)

    print(json.dumps(display_results, indent=2, ensure_ascii=False))
    return results


def _parse_form_id(value: str) -> int | None:
    """Treat a blank form ID as a request to search the whole database."""
    if not value.strip():
        return None
    try:
        form_id = int(value)
    except ValueError as error:
        raise ValueError("form_id must be a positive integer or blank") from error
    if form_id < 1:
        raise ValueError("form_id must be a positive integer or blank")
    return form_id


def main() -> None:
    """Prompt for query, model, optional form ID, then print matching rows."""
    query_text = input("Enter your query (English or Hindi): ").strip()
    print("Select embedding model:")
    for option, (display_name, _) in MODEL_OPTIONS.items():
        print(f"{option}. {display_name}")
    model_choice = input("Enter model number (1-3): ").strip()
    if model_choice not in MODEL_OPTIONS:
        raise ValueError("Choose a model by entering 1, 2, or 3")
    model_name = MODEL_OPTIONS[model_choice][1]
    form_id = _parse_form_id(input("Enter form ID (blank to search all forms): "))
    resolve_query(
        query_text,
        form_id=form_id,
        top_k=10,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
