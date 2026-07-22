import re
from typing import Any


def tokenize(text: str) -> set[str]:
    """
    Convert text into lowercase word tokens.
    """
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def rerank_results(
    question: str,
    results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Improve vector-search ordering using simple keyword overlap.
    """

    question_tokens = tokenize(question)

    reranked = []

    for result in results:
        score = result["score"]

        title_tokens = tokenize(result["title"])
        section_tokens = tokenize(result["section"])
        content_tokens = tokenize(result["content"])

        title_overlap = len(question_tokens & title_tokens)
        section_overlap = len(question_tokens & section_tokens)
        content_overlap = len(question_tokens & content_tokens)

        score += title_overlap * 0.20
        score += section_overlap * 0.10
        score += content_overlap * 0.01

        new_result = result.copy()
        new_result["rerank_score"] = score

        reranked.append(new_result)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]