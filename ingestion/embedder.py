import json
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def load_chunks(file_path: str | Path) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_embedding_text(chunk: dict) -> str:
    return (
        f"Section {chunk['section']}: {chunk['title']}\n"
        f"{chunk['text']}"
    )


def generate_embeddings(
    chunks: list[dict],
) -> list[list[float]]:
    model = get_embedding_model()

    texts = [
        build_embedding_text(chunk)
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    return embedding.tolist()


def main() -> None:
    chunk_path = Path("data/processed/ts23501_chunks.json")

    chunks = load_chunks(chunk_path)

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    embeddings = generate_embeddings(chunks)

    print(f"Embeddings created: {len(embeddings)}")

    if embeddings:
        print(f"Embedding dimensions: {len(embeddings[0])}")
        print(f"First five values: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()