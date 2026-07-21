import json
from pathlib import Path

from database.connection import get_connection
from ingestion.embedder import generate_embeddings, load_chunks


CHUNK_PATH = Path("data/processed/ts23501_chunks.json")


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

    with get_connection() as connection:
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

    print(f"Rows inserted: {len(rows)}")


if __name__ == "__main__":
    main()