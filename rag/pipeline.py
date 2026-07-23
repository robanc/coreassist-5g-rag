from functools import lru_cache
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, RETRIEVAL_LIMIT
from retrieval.reranker import search_and_rerank


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


REWRITE_PROMPT = """
You rewrite follow-up questions into standalone questions.

Rules:
- Use the conversation history.
- Replace pronouns like "it", "they", "this", and "that" with the
  specific entity they refer to.
- Preserve the user's original intent.
- Do not answer the question.
- Return only the rewritten standalone question.
- If the question is already standalone, return it unchanged.
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
    """
    Convert reranked chunks into a structured context block for the LLM.
    """
    context_blocks: list[str] = []

    for result in results:
        context_blocks.append(
            "\n".join(
                [
                    f"Section: {result['section']}",
                    f"Title: {result['title']}",
                    f"Source: {result['source']}",
                    (
                        f"Vector score: "
                        f"{result['vector_score']:.6f}"
                    ),
                    (
                        f"Rerank score: "
                        f"{result['rerank_score']:.6f}"
                    ),
                    "Content:",
                    result["content"],
                ]
            )
        )

    return "\n\n---\n\n".join(context_blocks)


def rewrite_question(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Rewrite a conversational follow-up into a standalone retrieval query.
    """
    if not conversation_history:
        return question

    client = get_openai_client()

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": REWRITE_PROMPT,
        }
    ]

    messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=messages,
    )

    rewritten_question = response.choices[0].message.content

    if not rewritten_question:
        return question

    return rewritten_question.strip()


def answer_question(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Rewrite the question, retrieve candidate chunks using vector search,
    rerank them with a cross-encoder, and generate a grounded answer.
    """
    standalone_question = rewrite_question(
        question=question,
        conversation_history=conversation_history,
    )

    print(f"\nOriginal question:  {question}")
    print(f"Rewritten question: {standalone_question}\n")

    results = search_and_rerank(
        query=standalone_question,
        limit=limit,
        candidate_limit=30,
    )

    if not results:
        return {
            "question": question,
            "standalone_question": standalone_question,
            "answer": (
                "I could not find relevant excerpts in the indexed "
                "3GPP TS 23.501 content."
            ),
            "sources": [],
        }

    print("Top reranked results:")

    for result in results:
        print(
            f"{result['rerank_score']:.6f} | "
            f"{result['section']} | "
            f"{result['title']} | "
            f"vector_score={result['vector_score']:.6f}"
        )

    print()

    context = build_context(results)
    client = get_openai_client()

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Standalone retrieval question:\n"
                f"{standalone_question}\n\n"
                f"3GPP excerpts:\n{context}"
            ),
        }
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=messages,
    )

    answer = response.choices[0].message.content

    if not answer:
        answer = "The model did not return an answer."

    return {
        "question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": results,
    }


def main() -> None:
    question = "What is the role of the AMF?"

    result = answer_question(question)

    print(f"\nQuestion:\n{result['question']}\n")
    print(
        "Retrieval question:\n"
        f"{result['standalone_question']}\n"
    )

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
            f"(rerank={source['rerank_score']:.6f}, "
            f"vector={source['vector_score']:.6f})"
        )


if __name__ == "__main__":
    main()