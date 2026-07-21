from typing import Any

from pgvector import Vector
from psycopg.rows import dict_row

from config import RETRIEVAL_LIMIT
from database.connection import get_connection
from ingestion.embedder import embed_query


SEARCH_SQL = """
SELECT
    id,
    source,
    title,
    section,
    content,
    metadata,
    1 - (embedding <=> %s) AS score
FROM documents
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s
LIMIT %s
"""


def search_documents(
    query: str,
    limit: int = RETRIEVAL_LIMIT,
) -> list[dict[str, Any]]:
    query_embedding = Vector(embed_query(query))

    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                SEARCH_SQL,
                (
                    query_embedding,
                    query_embedding,
                    limit,
                ),
            )

            return list(cursor.fetchall())


def main() -> None:
    query = "What is the role of the AMF?"
    results = search_documents(query)

    print(f"\nQuery: {query}\n")

    for index, result in enumerate(results, start=1):
        print(f"Result {index}")
        print(f"Score: {result['score']:.4f}")
        print(f"Section: {result['section']}")
        print(f"Title: {result['title']}")
        print(result["content"])
        print("-" * 80)


if __name__ == "__main__":
    main()