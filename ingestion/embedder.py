import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(file_path: str | Path) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_embeddings(
    chunks: list[dict],
    model_name: str = MODEL_NAME,
) -> list[list[float]]:
    model = SentenceTransformer(model_name)

    texts = [
    f"Section {chunk['section']}: {chunk['title']}\n{chunk['text']}"
    for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def main() -> None:
    chunk_path = Path("data/processed/ts23501_chunks.json")

    chunks = load_chunks(chunk_path)

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Embedding model: {MODEL_NAME}")

    embeddings = generate_embeddings(chunks)

    print(f"Embeddings created: {len(embeddings)}")

    if embeddings:
        print(f"Embedding dimensions: {len(embeddings[0])}")
        print(f"First five values: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()