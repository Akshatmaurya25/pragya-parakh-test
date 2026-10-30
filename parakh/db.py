"""PostgreSQL helpers. One short-lived connection per call."""

from typing import Any

import psycopg2

from parakh import config


def connect():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASS,
    )


def fetch(query: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    """Run a query and return its rows as dicts (empty list for statements without rows)."""
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description is None:
                return []
            columns = [column.name for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def execute_many(query: str, params_list: list[tuple[Any, ...]]) -> int:
    """Run a parameterised write for each parameter set, commit, return affected rows."""
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(query, params_list)
            row_count = cursor.rowcount
        connection.commit()
        return row_count
    finally:
        connection.close()
