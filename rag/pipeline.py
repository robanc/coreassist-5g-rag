import os

from openai import OpenAI

from retrieval.search import search_documents


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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


def build_context(results: list[dict]) -> str:
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
    limit: int = 5,
) -> dict:
    results = search_documents(question, limit=limit)
    context = build_context(results)

    client = OpenAI()

    response = client.chat.completions.create(
        model=MODEL_NAME,
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

    for source in result["sources"]:
        print(
            f"- Section {source['section']}: "
            f"{source['title']} "
            f"(score={source['score']:.4f})"
        )


if __name__ == "__main__":
    main()