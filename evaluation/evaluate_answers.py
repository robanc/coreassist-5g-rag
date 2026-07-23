import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_JUDGE_MODEL,
)

from rag.pipeline import answer_question, build_context


QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.csv"
RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"

JUDGE_SYSTEM_PROMPT = """
You are an impartial evaluator of answers produced by a 5G Packet Core
retrieval-augmented generation system.

Evaluate the generated answer using only:
1. The user's question.
2. The retrieved excerpts from 3GPP TS 23.501.

Do not reward unsupported outside knowledge.

Do not assume an answer is excellent merely because it includes citations.

Actively identify:
- claims not explicitly supported by the excerpts,
- omitted qualifications or conditions,
- irrelevant retrieved details repeated in the answer,
- citations that do not support the associated claim,
- answers that overstate what the excerpts establish.

A score of 5 should be rare and reserved for answers with no meaningful
factual, grounding, relevance, or completeness issues.
When uncertain between two scores, choose the lower score.

Score each criterion from 1 to 5:

Correctness:
1 = substantially incorrect
2 = major errors
3 = partly correct
4 = correct with minor issues
5 = fully correct according to the excerpts

Groundedness:
1 = mostly unsupported or contradicted
2 = several unsupported claims
3 = partially supported
4 = almost entirely supported
5 = every material claim is supported by the excerpts

Relevance:
1 = does not answer the question
2 = mostly irrelevant
3 = partially answers it
4 = directly answers it with minor digressions
5 = focused and directly responsive

Completeness:
1 = misses nearly all important information
2 = misses major points
3 = covers some important points
4 = covers most important points
5 = covers all important points supported by the excerpts

Return valid JSON only. Do not use Markdown or code fences.

Use exactly this structure:
{
  "correctness": 1,
  "groundedness": 1,
  "relevance": 1,
  "completeness": 1,
  "reasoning": "Brief explanation of the scores."
}
""".strip()


def get_openai_client() -> OpenAI:
    """Create an OpenAI client for answer judging."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your .env file."
        )

    return OpenAI(api_key=OPENAI_API_KEY)


def load_questions() -> list[dict[str, str]]:
    """Load the answer-evaluation questions."""
    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        questions = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required_columns = {
        "question",
        "expected_section",
        "category",
    }

    if not required_columns.issubset(fieldnames):
        raise ValueError(
            "Invalid questions.csv header. "
            f"Expected columns: {sorted(required_columns)}. "
            f"Found: {sorted(fieldnames)}"
        )

    if not questions:
        raise ValueError("The evaluation dataset is empty.")

    return questions


def extract_source_metadata(
    sources: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Extract section numbers and titles from retrieved sources."""
    sections = [
        str(source.get("section", "")).strip()
        for source in sources
    ]

    titles = [
        str(source.get("title", "")).strip()
        for source in sources
    ]

    return sections, titles


def parse_judge_response(content: str) -> dict[str, Any]:
    """Parse and validate the judge's JSON response."""
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The judge returned invalid JSON: {content}"
        ) from exc

    required_fields = {
        "correctness",
        "groundedness",
        "relevance",
        "completeness",
        "reasoning",
    }

    missing_fields = required_fields.difference(result)

    if missing_fields:
        raise ValueError(
            "Judge response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    for field in (
        "correctness",
        "groundedness",
        "relevance",
        "completeness",
    ):
        score = result[field]

        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(
                f"Invalid {field} score: {score}. "
                "Expected an integer from 1 to 5."
            )

    result["reasoning"] = str(result["reasoning"]).strip()

    return result


def judge_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use an LLM judge to evaluate one generated answer."""
    client = get_openai_client()
    context = build_context(sources)

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Generated answer:\n{answer}\n\n"
        f"Retrieved 3GPP excerpts:\n{context}"
    )

    response = client.chat.completions.create(
        model=OPENAI_JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The judge returned an empty response.")

    return parse_judge_response(content)


def run_judge_sanity_checks() -> None:
    """Verify that the judge can distinguish strong and weak answers."""
    test_cases = [
        {
            "label": "unsupported",
            "question": "What is the role of the AMF?",
            "answer": (
                "The AMF encrypts all user-plane traffic and stores "
                "subscriber billing records."
            ),
            "expected_maximum": 2.5,
        },
        {
            "label": "irrelevant",
            "question": "What is the role of the AMF?",
            "answer": "The UPF forwards user-plane packets.",
            "expected_maximum": 2.5,
        },
        {
            "label": "incomplete",
            "question": "What is the role of the AMF?",
            "answer": "The AMF handles registration.",
            "expected_maximum": 4.0,
        },
    ]

    rag_result = answer_question("What is the role of the AMF?")
    sources = rag_result["sources"]

    print("\nJudge sanity checks")
    print("-------------------")

    for test_case in test_cases:
        judge_result = judge_answer(
            question=test_case["question"],
            answer=test_case["answer"],
            sources=sources,
        )

        scores = [
            judge_result["correctness"],
            judge_result["groundedness"],
            judge_result["relevance"],
            judge_result["completeness"],
        ]

        average_score = sum(scores) / len(scores)

        print(
            f"{test_case['label']}: "
            f"average={average_score:.2f} | "
            f"{judge_result['reasoning']}"
        )

        if average_score > test_case["expected_maximum"]:
            raise RuntimeError(
                f"Judge sanity check failed for "
                f"{test_case['label']}: {average_score:.2f}"
            )


def evaluate_question(
    item: dict[str, str],
) -> dict[str, Any]:
    """Generate and judge an answer for one evaluation question."""
    question = item["question"].strip()

    rag_result = answer_question(
        question=question,
        conversation_history=None,
    )

    answer = str(rag_result.get("answer", "")).strip()
    sources = rag_result.get("sources", [])

    if not isinstance(sources, list):
        raise TypeError("The RAG pipeline returned invalid sources.")

    judge_result = judge_answer(
        question=question,
        answer=answer,
        sources=sources,
    )

    sections, titles = extract_source_metadata(sources)

    score_values = [
        judge_result["correctness"],
        judge_result["groundedness"],
        judge_result["relevance"],
        judge_result["completeness"],
    ]

    average_score = sum(score_values) / len(score_values)

    return {
        "question": question,
        "category": item.get("category", ""),
        "expected_section": item.get("expected_section", ""),
        "standalone_question": rag_result.get(
            "standalone_question",
            question,
        ),
        "answer": answer,
        "retrieved_sections": sections,
        "retrieved_titles": titles,
        "correctness": judge_result["correctness"],
        "groundedness": judge_result["groundedness"],
        "relevance": judge_result["relevance"],
        "completeness": judge_result["completeness"],
        "average_score": average_score,
        "judge_reasoning": judge_result["reasoning"],
    }


def calculate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate aggregate answer-quality metrics."""
    if not results:
        raise ValueError("No answer-evaluation results were produced.")

    number_of_questions = len(results)

    def average(field: str) -> float:
        return (
            sum(float(result[field]) for result in results)
            / number_of_questions
        )

    passing_results = [
        result
        for result in results
        if float(result["average_score"]) >= 4.0
    ]

    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "number_of_questions": number_of_questions,
        "average_correctness": average("correctness"),
        "average_groundedness": average("groundedness"),
        "average_relevance": average("relevance"),
        "average_completeness": average("completeness"),
        "overall_average_score": average("average_score"),
        "passing_threshold": 4.0,
        "passing_questions": len(passing_results),
        "pass_rate": len(passing_results) / number_of_questions,
    }


def save_results(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Save detailed answer results and summary metrics."""
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    detailed_path = (
        RESULTS_DIRECTORY / f"answer_results_{timestamp}.csv"
    )
    metrics_path = (
        RESULTS_DIRECTORY / f"answer_metrics_{timestamp}.json"
    )

    fieldnames = [
        "question",
        "category",
        "expected_section",
        "standalone_question",
        "answer",
        "retrieved_sections",
        "retrieved_titles",
        "correctness",
        "groundedness",
        "relevance",
        "completeness",
        "average_score",
        "judge_reasoning",
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

            output_row["average_score"] = (
                f"{result['average_score']:.3f}"
            )

            writer.writerow(output_row)

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print(f"\nDetailed results: {detailed_path}")
    print(f"Summary metrics:  {metrics_path}")


def main() -> None:
    """Run the answer-quality evaluation suite."""
    questions = load_questions()

    run_judge_sanity_checks()

    evaluation_results: list[dict[str, Any]] = []

    print(
        f"Evaluating generated answers for "
        f"{len(questions)} questions...\n"
    )

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

        print(
            "  Scores: "
            f"correctness={result['correctness']}, "
            f"groundedness={result['groundedness']}, "
            f"relevance={result['relevance']}, "
            f"completeness={result['completeness']}, "
            f"average={result['average_score']:.2f}"
        )

        print(f"  Judge: {result['judge_reasoning']}")

    if not evaluation_results:
        raise RuntimeError(
            "No questions were evaluated successfully."
        )

    metrics = calculate_metrics(evaluation_results)

    print("\nAnswer-quality evaluation summary")
    print("---------------------------------")
    print(
        f"Questions:             "
        f"{metrics['number_of_questions']}"
    )
    print(
        f"Correctness:           "
        f"{metrics['average_correctness']:.3f}"
    )
    print(
        f"Groundedness:          "
        f"{metrics['average_groundedness']:.3f}"
    )
    print(
        f"Relevance:             "
        f"{metrics['average_relevance']:.3f}"
    )
    print(
        f"Completeness:          "
        f"{metrics['average_completeness']:.3f}"
    )
    print(
        f"Overall average:       "
        f"{metrics['overall_average_score']:.3f}"
    )
    print(
        f"Pass rate (>= 4.0):    "
        f"{metrics['pass_rate']:.3f}"
    )

    save_results(evaluation_results, metrics)


if __name__ == "__main__":
    main()