import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.reranker import search_and_rerank


QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.csv"
RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"


def normalize_section(section: Any) -> str:
    """Normalize a section number for comparison."""
    return str(section).strip().replace("§", "")


def section_matches(
    retrieved_section: str,
    expected_section: str,
) -> bool:
    """
    Compare retrieved and expected sections.

    A subsection counts as a match when it belongs to the expected
    section. For example, 5.6.1 matches an expected value of 5.6.
    """
    retrieved = normalize_section(retrieved_section)
    expected = normalize_section(expected_section)

    return retrieved == expected or retrieved.startswith(f"{expected}.")


def load_questions() -> list[dict[str, str]]:
    """Load the retrieval evaluation dataset."""
    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        questions = list(reader)

    required_columns = {
        "question",
        "expected_section",
        "category",
    }

    actual_columns = set(reader.fieldnames or [])

    if not required_columns.issubset(actual_columns):
        raise ValueError(
            "Invalid questions.csv header. "
            f"Expected columns: {sorted(required_columns)}. "
            f"Found: {sorted(actual_columns)}"
        )

    if not questions:
        raise ValueError("The evaluation dataset is empty.")

    return questions


def evaluate_question(
    item: dict[str, str],
    limit: int = 5,
    candidate_limit: int = 30,
) -> dict[str, Any]:
    """Evaluate retrieval for one question."""
    question = item["question"]
    expected_section = normalize_section(item["expected_section"])

    results = search_and_rerank(
        query=question,
        limit=limit,
        candidate_limit=candidate_limit,
    )

    retrieved_sections = [
        normalize_section(result.get("section", ""))
        for result in results
    ]

    retrieved_titles = [
        str(result.get("title", ""))
        for result in results
    ]

    retrieved_scores = [
        float(
            result.get(
                "rerank_score",
                result.get(
                    "score",
                    result.get("vector_score", 0.0),
                ),
            )
        )
        for result in results
    ]

    matching_ranks = [
        rank
        for rank, section in enumerate(retrieved_sections, start=1)
        if section_matches(section, expected_section)
    ]

    first_matching_rank = matching_ranks[0] if matching_ranks else None

    return {
        "question": question,
        "category": item.get("category", ""),
        "expected_section": expected_section,
        "retrieved_sections": retrieved_sections,
        "retrieved_titles": retrieved_titles,
        "retrieved_scores": retrieved_scores,
        "top_1_hit": first_matching_rank == 1,
        "top_3_hit": (
            first_matching_rank is not None
            and first_matching_rank <= 3
        ),
        "top_5_hit": (
            first_matching_rank is not None
            and first_matching_rank <= 5
        ),
        "first_matching_rank": first_matching_rank,
        "reciprocal_rank": (
            1 / first_matching_rank
            if first_matching_rank is not None
            else 0.0
        ),
    }


def calculate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate aggregate retrieval metrics."""
    total = len(results)

    if total == 0:
        raise ValueError("No successful evaluation results were produced.")

    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "number_of_questions": total,
        "hit_rate_at_1": (
            sum(result["top_1_hit"] for result in results) / total
        ),
        "hit_rate_at_3": (
            sum(result["top_3_hit"] for result in results) / total
        ),
        "hit_rate_at_5": (
            sum(result["top_5_hit"] for result in results) / total
        ),
        "mean_reciprocal_rank": (
            sum(result["reciprocal_rank"] for result in results) / total
        ),
    }


def save_results(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Save detailed and aggregate evaluation results."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    detailed_path = (
        RESULTS_DIRECTORY / f"retrieval_results_{timestamp}.csv"
    )
    metrics_path = (
        RESULTS_DIRECTORY / f"retrieval_metrics_{timestamp}.json"
    )

    fieldnames = [
        "question",
        "category",
        "expected_section",
        "retrieved_sections",
        "retrieved_titles",
        "retrieved_scores",
        "top_1_hit",
        "top_3_hit",
        "top_5_hit",
        "first_matching_rank",
        "reciprocal_rank",
    ]

    with detailed_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            output_row = result.copy()

            output_row["retrieved_sections"] = " | ".join(
                result["retrieved_sections"]
            )

            output_row["retrieved_titles"] = " | ".join(
                result["retrieved_titles"]
            )

            output_row["retrieved_scores"] = " | ".join(
                f"{score:.6f}"
                for score in result["retrieved_scores"]
            )

            writer.writerow(output_row)

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print(f"\nDetailed results: {detailed_path}")
    print(f"Summary metrics:  {metrics_path}")


def print_missed_results(
    result: dict[str, Any],
) -> None:
    """Print retrieved sections for a missed evaluation question."""
    print("  Retrieved:")

    for section, title, score in zip(
        result["retrieved_sections"],
        result["retrieved_titles"],
        result["retrieved_scores"],
        strict=True,
    ):
        print(
            f"    - {section} | {title} | "
            f"score={score:.6f}"
        )


def main() -> None:
    """Run the retrieval evaluation suite."""
    questions = load_questions()
    evaluation_results: list[dict[str, Any]] = []

    print(f"Evaluating {len(questions)} questions...\n")

    for index, item in enumerate(questions, start=1):
        question = item.get("question", "").strip()

        if not question:
            print(f"[{index}/{len(questions)}] Skipped empty question")
            continue

        print(f"[{index}/{len(questions)}] {question}")

        try:
            result = evaluate_question(item)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        evaluation_results.append(result)

        rank = result["first_matching_rank"]
        status = f"rank {rank}" if rank else "not found"

        print(
            f"  Expected: {result['expected_section']} | "
            f"Result: {status}"
        )

        if rank is None:
            print_missed_results(result)

    if not evaluation_results:
        raise RuntimeError("No questions were evaluated successfully.")

    metrics = calculate_metrics(evaluation_results)

    print("\nRetrieval evaluation summary")
    print("--------------------------------")
    print(
        f"Questions:  {metrics['number_of_questions']}"
    )
    print(
        f"Hit Rate@1: {metrics['hit_rate_at_1']:.3f}"
    )
    print(
        f"Hit Rate@3: {metrics['hit_rate_at_3']:.3f}"
    )
    print(
        f"Hit Rate@5: {metrics['hit_rate_at_5']:.3f}"
    )
    print(
        f"MRR:        {metrics['mean_reciprocal_rank']:.3f}"
    )

    save_results(evaluation_results, metrics)


if __name__ == "__main__":
    main()