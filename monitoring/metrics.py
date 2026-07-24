from typing import Any

import pandas as pd

from database.connection import get_connection


def create_tables() -> None:
    """
    Create monitoring tables if they do not already exist.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_requests (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

                    question TEXT NOT NULL,
                    standalone_question TEXT,

                    model TEXT,

                    response_time_ms DOUBLE PRECISION,

                    retrieved_documents INTEGER,

                    top_rerank_score DOUBLE PRECISION,
                    top_vector_score DOUBLE PRECISION,

                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,

                    context_length INTEGER
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_feedback (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

                    request_id INTEGER NOT NULL UNIQUE
                        REFERENCES rag_requests(id)
                        ON DELETE CASCADE,

                    helpful BOOLEAN NOT NULL,
                    comments TEXT
                );
                """
            )

            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    rag_feedback_request_id_idx
                ON rag_feedback (request_id);
                """
            )

        conn.commit()


def log_request(data: dict[str, Any]) -> int:
    """
    Store one RAG request and return its database ID.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO rag_requests (

                    question,
                    standalone_question,

                    model,

                    response_time_ms,

                    retrieved_documents,

                    top_rerank_score,
                    top_vector_score,

                    prompt_tokens,
                    completion_tokens,
                    total_tokens,

                    context_length

                )
                VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                )
                RETURNING id;
                """,
                (
                    data["question"],
                    data["standalone_question"],
                    data["model"],
                    data["response_time_ms"],
                    data["retrieved_documents"],
                    data["top_rerank_score"],
                    data["top_vector_score"],
                    data["prompt_tokens"],
                    data["completion_tokens"],
                    data["total_tokens"],
                    data["context_length"],
                ),
            )

            request_id = cur.fetchone()[0]

        conn.commit()

    return request_id


def save_feedback(
    request_id: int,
    helpful: bool,
    comments: str | None = None,
) -> None:
    """
    Insert or update user feedback for a request.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO rag_feedback (
                    request_id,
                    helpful,
                    comments
                )
                VALUES (%s, %s, %s)

                ON CONFLICT (request_id)

                DO UPDATE SET
                    helpful = EXCLUDED.helpful,
                    comments = EXCLUDED.comments,
                    created_at = NOW();
                """,
                (
                    request_id,
                    helpful,
                    comments or None,
                ),
            )

        conn.commit()


def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load monitoring data into pandas DataFrames.
    """

    with get_connection() as conn:

        requests = pd.read_sql_query(
            """
            SELECT *
            FROM rag_requests
            ORDER BY created_at DESC;
            """,
            conn,
        )

        feedback = pd.read_sql_query(
            """
            SELECT *
            FROM rag_feedback
            ORDER BY created_at DESC;
            """,
            conn,
        )

    return requests, feedback