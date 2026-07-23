from functools import lru_cache
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, RETRIEVAL_LIMIT
from monitoring.tracing import get_tracer
from retrieval.reranker import search_and_rerank


tracer = get_tracer()


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
    Run the complete CoreAssist RAG pipeline inside an OpenTelemetry trace.
    """
    with tracer.start_as_current_span(
        "rag.answer_question"
    ) as span:
        span.set_attribute("rag.question", question)
        span.set_attribute("rag.retrieval_limit", limit)
        span.set_attribute(
            "rag.has_conversation_history",
            bool(conversation_history),
        )

        result = _answer_question(
            question=question,
            limit=limit,
            conversation_history=conversation_history,
        )

        span.set_attribute(
            "rag.retrieved_document_count",
            len(result["sources"]),
        )
        span.set_attribute(
            "rag.answer_length",
            len(result["answer"]),
        )

        return result


def _answer_question(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Rewrite the question, retrieve candidate chunks using vector search,
    rerank them with a cross-encoder, and generate a grounded answer.
    """
    with tracer.start_as_current_span(
        "rag.query_rewrite"
    ) as span:
        span.set_attribute(
            "rag.has_conversation_history",
            bool(conversation_history),
        )

        standalone_question = rewrite_question(
            question=question,
            conversation_history=conversation_history,
        )

        span.set_attribute(
            "rag.question_was_rewritten",
            standalone_question != question,
        )

    print(f"\nOriginal question:  {question}")
    print(f"Rewritten question: {standalone_question}\n")

    with tracer.start_as_current_span(
        "rag.retrieval_and_reranking"
    ) as span:
        span.set_attribute(
            "rag.retrieval_query",
            standalone_question,
        )
        span.set_attribute("rag.candidate_limit", 30)
        span.set_attribute("rag.result_limit", limit)

        results = search_and_rerank(
            query=standalone_question,
            limit=limit,
            candidate_limit=30,
        )

        span.set_attribute("rag.result_count", len(results))

        if results:
            span.set_attribute(
                "rag.top_rerank_score",
                float(results[0]["rerank_score"]),
            )
            span.set_attribute(
                "rag.top_vector_score",
                float(results[0]["vector_score"]),
            )
            span.set_attribute(
                "rag.top_section",
                str(results[0]["section"]),
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

    with tracer.start_as_current_span(
        "rag.context_building"
    ) as span:
        context = build_context(results)

        span.set_attribute(
            "rag.context_length",
            len(context),
        )
        span.set_attribute(
            "rag.context_chunk_count",
            len(results),
        )

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

    with tracer.start_as_current_span(
        "rag.answer_generation"
    ) as span:
        span.set_attribute("llm.model", OPENAI_MODEL)
        span.set_attribute("llm.temperature", 0)
        span.set_attribute(
            "llm.message_count",
            len(messages),
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=messages,
        )

        answer = response.choices[0].message.content

        span.set_attribute(
            "llm.answer_length",
            len(answer) if answer else 0,
        )

        if response.usage:
            span.set_attribute(
                "llm.input_tokens",
                response.usage.prompt_tokens,
            )
            span.set_attribute(
                "llm.output_tokens",
                response.usage.completion_tokens,
            )
            span.set_attribute(
                "llm.total_tokens",
                response.usage.total_tokens,
            )

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