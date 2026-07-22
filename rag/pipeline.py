from functools import lru_cache
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, RETRIEVAL_LIMIT
from rag.reranker import rerank_results
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
    context_blocks = []

    for result in results:
        context_blocks.append(
            "\n".join(
                [
                    f"Section: {result['section']}",
                    f"Title: {result['title']}",
                    f"Source: {result['source']}",
                    f"Vector similarity score: {result['score']:.4f}",
                    (
                        f"Rerank score: "
                        f"{result.get('rerank_score', result['score']):.4f}"
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
    standalone_question = rewrite_question(
        question=question,
        conversation_history=conversation_history,
    )

    print(f"\nOriginal question:  {question}")
    print(f"Rewritten question: {standalone_question}\n")

    candidate_limit = max(limit * 2, 10)

    candidate_results = search_documents(
        standalone_question,
        limit=candidate_limit,
    )

    if not candidate_results:
        return {
            "question": question,
            "standalone_question": standalone_question,
            "answer": (
                "I could not find relevant excerpts in the indexed "
                "3GPP TS 23.501 content."
            ),
            "sources": [],
        }

    results = rerank_results(
        question=standalone_question,
        results=candidate_results,
        top_k=limit,
    )

    print("Top reranked results:")

    for result in results:
        print(
            f"{result['rerank_score']:.4f} | "
            f"{result['section']} | "
            f"{result['title']} | "
            f"vector={result['score']:.4f}"
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
    print(f"Retrieval question:\n{result['standalone_question']}\n")

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
            f"(vector={source['score']:.4f}, "
            f"rerank={source.get('rerank_score', source['score']):.4f})"
        )


if __name__ == "__main__":
    main()