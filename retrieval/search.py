import os

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://coreassist:coreassist@localhost:5432/coreassist",
)


def search_documents(
    query: str,
    limit: int = 5,
) -> list[dict]:
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    with psycopg.connect(DATABASE_URL) as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                """
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
                """,
                (
                    query_embedding,
                    query_embedding,
                    limit,
                ),
            )

            rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "source": row[1],
            "title": row[2],
            "section": row[3],
            "content": row[4],
            "metadata": row[5],
            "score": float(row[6]),
        }
        for row in rows
    ]


def main() -> None:
    query = "What is the role of the AMF?"

    results = search_documents(query, limit=5)

    print(f"\nQuery: {query}\n")

    for index, result in enumerate(results, start=1):
        print(f"Result {index}")
        print(f"Score: {result['score']:.4f}")
        print(f"Section: {result['section']}")
        print(f"Title: {result['title']}")
        print(result["content"][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()