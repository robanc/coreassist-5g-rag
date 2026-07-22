import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import answer_question


st.set_page_config(
    page_title="CoreAssist",
    page_icon="📡",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📡 CoreAssist")
st.subheader("5G Packet Core Engineering Assistant")
st.caption("Ask questions grounded in 3GPP TS 23.501.")

with st.sidebar:
    st.header("About CoreAssist")

    st.write(
        "CoreAssist answers 5G Packet Core questions using "
        "retrieved excerpts from 3GPP TS 23.501."
    )

    st.markdown("**Knowledge source**")
    st.write("3GPP TS 23.501 Release 19")

    st.markdown("**Retrieval**")
    st.write("PostgreSQL + pgvector")

    st.markdown("**Embedding model**")
    st.write("all-MiniLM-L6-v2")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "📡"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(
                f"Retrieved sources ({len(message['sources'])})"
            ):
                for source in message["sources"]:
                    st.markdown(
                        f"**§{source['section']} — {source['title']}**  \n"
                        f"Similarity score: `{source['score']:.3f}`"
                    )
                    st.write(source["content"])
                    st.divider()

question = st.chat_input(
    "Ask a question about 3GPP TS 23.501"
)

if question:
    conversation_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages[-6:]
    ]

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📡"):
        with st.spinner("Searching the specification..."):
            try:
                result = answer_question(
                    question,
                    conversation_history=conversation_history,
                )
            except Exception as exc:
                error_message = f"Unable to answer the question: {exc}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )

                st.stop()

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(
                f"Retrieved sources ({len(result['sources'])})"
            ):
                for source in result["sources"]:
                    st.markdown(
                        f"**§{source['section']} — {source['title']}**  \n"
                        f"Similarity score: `{source['score']:.3f}`"
                    )
                    st.write(source["content"])
                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )