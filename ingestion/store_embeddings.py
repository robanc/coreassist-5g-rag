import json
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from ingestion.embedder import generate_embeddings, load_chunks


CHUNK_PATH = Path("data/processed/ts23501_chunks.json")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://coreassist:coreassist@localhost:5432/coreassist",
)


def main() -> None:
    chunks = load_chunks(CHUNK_PATH)

    print(f"Chunks loaded: {len(chunks)}")

    embeddings = generate_embeddings(chunks)

    rows = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        metadata = {
            "spec": chunk["spec"],
            "release": chunk["release"],
            "version": chunk["version"],
            "chunk_id": chunk["chunk_id"],
        }

        rows.append(
            (
                chunk["source"],
                chunk["title"],
                chunk["section"],
                chunk["text"],
                json.dumps(metadata),
                embedding,
            )
        )

    with psycopg.connect(DATABASE_URL) as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM documents
                WHERE source = %s
                """,
                ("TS23501.docx",),
            )

            cursor.executemany(
                """
                INSERT INTO documents (
                    source,
                    title,
                    section,
                    content,
                    metadata,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                rows,
            )

        connection.commit()

    print(f"Rows inserted: {len(rows)}")


if __name__ == "__main__":
    main()