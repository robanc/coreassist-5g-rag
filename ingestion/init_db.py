import os

import psycopg
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://coreassist:coreassist@localhost:5432/coreassist",
)


def initialize_database() -> None:
    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    source VARCHAR(500) NOT NULL,
                    title VARCHAR(500),
                    section VARCHAR(500),
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding VECTOR(384),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents
                USING hnsw (embedding vector_cosine_ops);
                """
            )

        connection.commit()

    print("Database initialized successfully.")


if __name__ == "__main__":
    initialize_database()