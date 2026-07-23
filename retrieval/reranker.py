from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from config import RETRIEVAL_LIMIT
from retrieval.search import search_documents


RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_CANDIDATE_LIMIT = 30


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Load and cache the cross-encoder reranker.

    The model is loaded only once within each Python process.
    """
    return CrossEncoder(
        model_name_or_path=RERANKER_MODEL,
        max_length=512,
    )


def build_passage(result: dict[str, Any]) -> str:
    """
    Combine a retrieved chunk's section, title, and content into one
    passage for cross-encoder scoring.
    """
    return "\n".join(
        [
            f"Section: {result.get('section') or ''}",
            f"Title: {result.get('title') or ''}",
            f"Content: {result.get('content') or ''}",
        ]
    )


def deduplicate_results(
    results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Remove duplicate section-title combinations after reranking.

    Long specification sections may be split into multiple chunks.
    This keeps only the highest-ranked chunk from each section and title.
    """
    unique_results: list[dict[str, Any]] = []
    seen_sections: set[tuple[str, str]] = set()

    for result in results:
        key = (
            str(result.get("section") or ""),
            str(result.get("title") or ""),
        )

        if key in seen_sections:
            continue

        seen_sections.add(key)
        unique_results.append(result)

        if len(unique_results) >= top_k:
            break

    return unique_results


def rerank_results(
    question: str,
    results: list[dict[str, Any]],
    top_k: int = RETRIEVAL_LIMIT,
) -> list[dict[str, Any]]:
    """
    Rerank vector-search candidates using a cross-encoder.

    Each candidate is scored together with the user's question.
    Results are sorted by reranker score and deduplicated by section
    and title.
    """
    if top_k <= 0 or not results:
        return []

    reranker = get_reranker()

    pairs = [
        [
            question,
            build_passage(result),
        ]
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    reranked_results: list[dict[str, Any]] = []

    for result, score in zip(results, scores, strict=True):
        reranked_result = result.copy()

        reranked_result["vector_score"] = float(
            result.get("score", 0.0)
        )
        reranked_result["rerank_score"] = float(score)

        # Keep the generic score field for compatibility with the UI
        # and other parts of the application.
        reranked_result["score"] = float(score)

        reranked_results.append(reranked_result)

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return deduplicate_results(
        results=reranked_results,
        top_k=top_k,
    )


def search_and_rerank(
    query: str,
    limit: int = RETRIEVAL_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """
    Retrieve candidate chunks using pgvector and rerank them using the
    cross-encoder.

    The candidate limit must be at least as large as the final result
    limit.
    """
    if limit <= 0:
        return []

    candidate_limit = max(candidate_limit, limit)

    candidates = search_documents(
        query=query,
        limit=candidate_limit,
    )

    return rerank_results(
        question=query,
        results=candidates,
        top_k=limit,
    )


def main() -> None:
    """
    Run a standalone retrieval and reranking test.
    """
    question = "What is the role of the AMF?"

    results = search_and_rerank(
        query=question,
        limit=5,
        candidate_limit=30,
    )

    print(f"\nQuestion: {question}\n")
    print("Cross-encoder reranked results:")

    if not results:
        print("No results found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Section: {result['section']}")
        print(f"Title: {result['title']}")
        print(
            f"Vector score: "
            f"{result['vector_score']:.4f}"
        )
        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )
        print("Content:")
        print(result["content"][:700])
        print("-" * 80)


if __name__ == "__main__":
    main()