"""Search all configured embedding models and compare their results in tables."""

import json
from typing import Any

from db_conn import execute
from embedder import MODEL_OPTIONS, MODEL_STORAGE_NAMES, embedder


DEFAULT_TOP_K = 20
EMBEDDING_COLUMNS = {"embedding_en", "embedding_hi"}
HIDDEN_DISPLAY_COLUMNS = {
    "created_at",
    "updated_at",
    "model",
    "similarity",
    *EMBEDDING_COLUMNS,
}
DISPLAY_WIDTH_LIMIT = 44


def _to_pgvector_literal(vector: list[float]) -> str:
    """Convert an embedding to PostgreSQL's vector input format."""
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _build_search_query(form_id: int | None) -> str:
    """Build a query that searches one model and optionally one form."""
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


def retrieve_model_matches(
    query_text: str,
    model_name: str,
    form_id: int | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Embed and search the query against one selected model's stored vectors."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    if model_name not in MODEL_STORAGE_NAMES:
        raise ValueError(f"Unsupported embedding model: {model_name}")
    if form_id is not None and (
        not isinstance(form_id, int) or isinstance(form_id, bool) or form_id < 1
    ):
        raise ValueError("form_id must be a positive integer or None")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    query_vector = _to_pgvector_literal(
        embedder(query_text.strip(), model_name=model_name).tolist()
    )
    storage_model_name = MODEL_STORAGE_NAMES[model_name]
    query = _build_search_query(form_id)

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


def _format_cell(value: Any, width: int) -> str:
    """Convert a table value to one compact line that fits its column."""
    if value is None:
        text = ""
    elif isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = str(value).replace("\r", " ").replace("\n", " ")
    if len(text) > width:
        return text[: max(0, width - 1)] + "…"
    return text


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    """Print rows as a compact plain-text table."""
    print(f"\n{title}")
    if not rows:
        print("(No matches.)")
        return

    widths = [
        min(
            DISPLAY_WIDTH_LIMIT,
            max(len(header), *(len(_format_cell(row[index], DISPLAY_WIDTH_LIMIT))
                              for row in rows)),
        )
        for index, header in enumerate(headers)
    ]

    def format_row(values: list[Any]) -> str:
        return " | ".join(
            _format_cell(value, width).ljust(width)
            for value, width in zip(values, widths)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def _model_table_data(
    results: list[dict[str, Any]],
) -> tuple[list[str], list[list[Any]]]:
    """Make display data from each model's full rows, excluding vectors."""
    headers = [
        key
        for key in results[0]
        if key not in HIDDEN_DISPLAY_COLUMNS
    ] if results else []
    headers.insert(min(3, len(headers)), "similarity_percentage")

    table_rows: list[list[Any]] = []
    for result in results:
        values = [
            result.get(key)
            for key in headers
            if key != "similarity_percentage"
        ]
        similarity = result.get("similarity")
        values.insert(min(3, len(values)),
            f"{float(similarity) * 100:.2f}%"
            if similarity is not None
            else None
        )
        table_rows.append(values)
    return headers, table_rows


def _combined_results(
    model_results: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[list[Any]]]:
    """Combine each model's best returned score for each form/question."""
    model_names = [display_name for display_name, _ in MODEL_OPTIONS.values()]
    combined: dict[tuple[Any, Any], dict[str, Any]] = {}

    for display_name, model_name in MODEL_OPTIONS.values():
        for result in model_results[model_name]:
            key = (result.get("form_id"), result.get("ques_id"))
            item = combined.setdefault(
                key,
                {
                    "form_id": result.get("form_id"),
                    "ques_id": result.get("ques_id"),
                    "question_caption": result.get("question_caption"),
                    "question_caption_hindi": result.get("question_caption_hindi"),
                },
            )
            score = result.get("similarity")
            current_score = item.get(display_name)
            if score is not None and (
                current_score is None or float(score) > float(current_score)
            ):
                item[display_name] = float(score)

    headers = [
        "form_id",
        "ques_id",
        "question_caption",
        "question_caption_hindi",
        *[f"{name} %" for name in model_names],
    ]
    table_rows: list[list[Any]] = []
    for key in sorted(
        combined,
        key=lambda entry: (
            entry[0] is None,
            str(entry[0]),
            entry[1] is None,
            str(entry[1]),
        ),
    ):
        item = combined[key]
        values = [
            item["form_id"],
            item["ques_id"],
            item["question_caption"],
            item["question_caption_hindi"],
        ]
        for model_name in model_names:
            score = item.get(model_name)
            values.append(f"{score * 100:.2f}%" if score is not None else "—")
        table_rows.append(values)
    return headers, table_rows


def resolve_all_models(
    query_text: str,
    form_id: int | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, list[dict[str, Any]]]:
    """Search all supported models, print per-model and combined tables."""
    model_results: dict[str, list[dict[str, Any]]] = {}
    for display_name, model_name in MODEL_OPTIONS.values():
        print(f"\nSearching with {display_name}...")
        results = retrieve_model_matches(
            query_text,
            model_name=model_name,
            form_id=form_id,
            top_k=top_k,
        )
        model_results[model_name] = results
        headers, rows = _model_table_data(results)
        print_table(f"Top matches: {display_name}", headers, rows)

    combined_headers, combined_rows = _combined_results(model_results)
    print_table("Combined model comparison by question", combined_headers, combined_rows)
    return model_results


def _parse_form_id(value: str) -> int | None:
    """Treat blank form ID input as a request to search every form."""
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
    """Prompt for a query and optional form, then compare all three models."""
    query_text = input("Enter your query (English or Hindi): ").strip()
    form_id = _parse_form_id(input("Enter form ID (blank to search all forms): "))
    resolve_all_models(query_text, form_id=form_id)


if __name__ == "__main__":
    main()
