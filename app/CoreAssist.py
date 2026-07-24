import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.metrics import save_feedback
from rag.pipeline import answer_question


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CoreAssist | 5G Packet Core Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-top: -0.7rem;
            margin-bottom: 1.5rem;
        }

        .source-label {
            font-size: 0.85rem;
            color: #6b7280;
        }

        .feedback-title {
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.25rem;
        }

        .feedback-saved {
            font-size: 0.85rem;
            color: #16a34a;
            margin-top: 0.75rem;
        }

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
            text-align: center;
            color: #6b7280;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "submitted_feedback" not in st.session_state:
    st.session_state.submitted_feedback = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clear_chat() -> None:
    """
    Clear the current conversation and local feedback state.

    Previously submitted feedback remains stored in PostgreSQL.
    """
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.submitted_feedback = {}


def select_example(question: str) -> None:
    """Store an example question for processing after the rerun."""
    st.session_state.pending_question = question


def get_source_score(
    source: dict[str, Any],
) -> tuple[str, float | None]:
    """
    Return the most useful score available in a retrieved source.

    The fallback order keeps the UI compatible with older and newer
    pipeline response formats.
    """
    score_fields = (
        ("Rerank score", "rerank_score"),
        ("Similarity score", "score"),
        ("Vector score", "vector_score"),
    )

    for label, field in score_fields:
        value = source.get(field)

        if isinstance(value, (int, float)):
            return label, float(value)

    return "Score", None


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render retrieved specification excerpts consistently."""
    if not sources:
        return

    with st.expander(
        f"View retrieved evidence ({len(sources)} sources)",
        expanded=False,
    ):
        for index, source in enumerate(sources, start=1):
            section = source.get("section", "Unknown section")
            title = source.get("title", "Untitled section")
            content = source.get(
                "content",
                "No excerpt available.",
            )

            score_label, score = get_source_score(source)

            st.markdown(
                f"#### {index}. §{section} — {title}"
            )

            if score is not None:
                st.markdown(
                    f'<span class="source-label">'
                    f"{score_label}: <code>{score:.3f}</code>"
                    f"</span>",
                    unsafe_allow_html=True,
                )

            st.markdown(content)

            if index < len(sources):
                st.divider()


def render_feedback(
    message: dict[str, Any],
    message_index: int,
) -> None:
    """
    Render a feedback form for an assistant response.

    Feedback is linked to the request row created by the RAG pipeline.
    """
    request_id = message.get("request_id")

    if not isinstance(request_id, int):
        return

    submitted_feedback = st.session_state.submitted_feedback

    if request_id in submitted_feedback:
        feedback_label = submitted_feedback[request_id]

        st.markdown(
            (
                '<div class="feedback-saved">'
                f"✓ Feedback saved: {feedback_label}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="feedback-title">'
        "Was this answer helpful?"
        "</div>",
        unsafe_allow_html=True,
    )

    form_key = (
        f"feedback_form_{request_id}_{message_index}"
    )

    with st.form(form_key, clear_on_submit=False):
        rating = st.radio(
            "Answer rating",
            options=(
                "👍 Helpful",
                "👎 Not helpful",
            ),
            horizontal=True,
            label_visibility="collapsed",
            key=(
                f"feedback_rating_"
                f"{request_id}_{message_index}"
            ),
        )

        comments = st.text_area(
            "Optional comments",
            placeholder=(
                "Tell us what was useful or what could "
                "be improved."
            ),
            height=80,
            key=(
                f"feedback_comments_"
                f"{request_id}_{message_index}"
            ),
        )

        submitted = st.form_submit_button(
            "Submit feedback",
            use_container_width=False,
        )

    if submitted:
        helpful = rating == "👍 Helpful"

        try:
            save_feedback(
                request_id=request_id,
                helpful=helpful,
                comments=comments.strip() or None,
            )
        except Exception as exc:
            st.error(
                "The feedback could not be saved. "
                f"Details: {exc}"
            )
            return

        st.session_state.submitted_feedback[
            request_id
        ] = rating

        st.success("Thank you. Your feedback was saved.")
        st.rerun()


def render_message(
    message: dict[str, Any],
    message_index: int,
) -> None:
    """Render a saved chat message."""
    role = message.get("role", "assistant")
    avatar = "👤" if role == "user" else "📡"

    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))

        if role == "assistant":
            render_sources(message.get("sources", []))
            render_feedback(
                message=message,
                message_index=message_index,
            )


def process_question(question: str) -> None:
    """Run the RAG pipeline and display the response."""
    conversation_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages[-6:]
        if message.get("role") in {"user", "assistant"}
    ]

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📡"):
        with st.spinner(
            "Rewriting the query, searching the specification, "
            "and reranking evidence..."
        ):
            try:
                result = answer_question(
                    question,
                    conversation_history=conversation_history,
                )
            except Exception as exc:
                error_message = (
                    "CoreAssist was unable to complete the request. "
                    f"Details: {exc}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "request_id": None,
                    }
                )
                return

        answer = result.get(
            "answer",
            "No answer was returned by the pipeline.",
        )

        sources = result.get("sources", [])
        request_id = result.get("request_id")

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "request_id": request_id,
        }

        st.session_state.messages.append(
            assistant_message
        )

        assistant_message_index = (
            len(st.session_state.messages) - 1
        )

        st.markdown(answer)
        render_sources(sources)
        render_feedback(
            message=assistant_message,
            message_index=assistant_message_index,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📡 CoreAssist")
    st.caption(
        "AI engineering assistant for 5G Packet Core"
    )

    st.divider()

    st.markdown("### About")
    st.write(
        "CoreAssist answers technical questions using evidence "
        "retrieved from the 3GPP system architecture "
        "specification."
    )

    st.markdown("### Knowledge base")
    st.markdown(
        """
        **Specification**  
        3GPP TS 23.501

        **Release**  
        Release 19

        **Coverage**  
        5G System architecture and network functions
        """
    )

    st.markdown("### Retrieval pipeline")
    st.markdown(
        """
        1. Conversation-aware query rewriting  
        2. Semantic search with pgvector  
        3. Cross-encoder reranking  
        4. Evidence-grounded answer generation
        """
    )

    st.markdown("### Technology")
    st.markdown(
        """
        - PostgreSQL and pgvector
        - all-MiniLM-L6-v2 embeddings
        - BGE cross-encoder reranker
        - OpenAI language model
        - OpenTelemetry tracing
        - Streamlit
        """
    )

    st.divider()

    st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
        on_click=clear_chat,
    )

    st.caption(
        "Responses should be verified against the official "
        "specification before use in production engineering "
        "decisions."
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

st.title("📡 CoreAssist")

st.markdown(
    '<div class="hero-subtitle">'
    "A retrieval-augmented 5G Packet Core engineering "
    "assistant grounded in 3GPP TS 23.501 Release 19."
    "</div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.info(
        "Ask about 5G network functions, registration, "
        "mobility, sessions, roaming, slicing, or other "
        "architecture topics."
    )

    st.markdown("### Example questions")

    example_questions = [
        "What is the role of the AMF?",
        "How does the SMF interact with the UPF?",
        "What are the main functions of the NRF?",
        "How does network slicing work in the 5G System?",
    ]

    first_row = st.columns(2)
    second_row = st.columns(2)

    for column, example in zip(
        first_row + second_row,
        example_questions,
        strict=True,
    ):
        with column:
            st.button(
                example,
                use_container_width=True,
                key=f"example_{example}",
                on_click=select_example,
                args=(example,),
            )

for message_index, saved_message in enumerate(
    st.session_state.messages
):
    render_message(
        message=saved_message,
        message_index=message_index,
    )

typed_question = st.chat_input(
    "Ask a question about 3GPP TS 23.501..."
)

question = (
    typed_question
    or st.session_state.pending_question
)

if question:
    st.session_state.pending_question = None
    process_question(question)

st.markdown(
    """
    <div class="footer">
        CoreAssist · Built with Streamlit, PostgreSQL,
        pgvector, sentence-transformers, OpenTelemetry,
        and OpenAI
    </div>
    """,
    unsafe_allow_html=True,
)