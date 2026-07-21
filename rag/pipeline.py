from functools import lru_cache
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, RETRIEVAL_LIMIT
from retrieval.search import search_documents


SYSTEM_PROMPT = """
You are CoreAssist, a 5G Packet Core engineering assistant.

Answer the user's question using only the provided excerpts from
3GPP TS 23.501.

Rules:
- Do not use unsupported outside knowledge.
- If the excerpts are insufficient, clearly say so.
- Give a concise technical explanation.
- Cite relevant sections using this format: [TS 23.501 §section].
""".strip()


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your .env file."
        )

    return OpenAI(api_key=OPENAI_API_KEY)


def build_context(results: list[dict[str, Any]]) -> str:
    context_blocks = []

    for result in results:
        context_blocks.append(
            "\n".join(
                [
                    f"Section: {result['section']}",
                    f"Title: {result['title']}",
                    f"Source: {result['source']}",
                    f"Similarity score: {result['score']:.4f}",
                    "Content:",
                    result["content"],
                ]
            )
        )

    return "\n\n---\n\n".join(context_blocks)


def answer_question(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
) -> dict[str, Any]:
    results = search_documents(question, limit=limit)

    if not results:
        return {
            "question": question,
            "answer": (
                "I could not find relevant excerpts in the indexed "
                "3GPP TS 23.501 content."
            ),
            "sources": [],
        }

    context = build_context(results)
    client = get_openai_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"3GPP excerpts:\n{context}"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        answer = "The model did not return an answer."

    return {
        "question": question,
        "answer": answer,
        "sources": results,
    }


def main() -> None:
    question = "What is the role of the AMF?"

    result = answer_question(question)

    print(f"\nQuestion:\n{result['question']}\n")
    print("Answer:")
    print(result["answer"])

    print("\nRetrieved sources:")

    if not result["sources"]:
        print("- No sources retrieved")
        return

    for source in result["sources"]:
        print(
            f"- Section {source['section']}: "
            f"{source['title']} "
            f"(score={source['score']:.4f})"
        )


if __name__ == "__main__":
    main()