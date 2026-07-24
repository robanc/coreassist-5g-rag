import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.metrics import load_dashboard_data


st.set_page_config(
    page_title="CoreAssist Monitoring",
    page_icon="📊",
    layout="wide",
)


st.title("📊 CoreAssist Monitoring Dashboard")
st.caption(
    "Operational metrics, retrieval quality, token usage, "
    "and user feedback."
)


requests_df, feedback_df = load_dashboard_data()


# ---------------------------------------------------------------------------
# Empty database
# ---------------------------------------------------------------------------

if requests_df.empty:
    st.info(
        "No monitoring data is available yet. Ask a question in "
        "CoreAssist to generate the first monitoring record."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------

requests_df["created_at"] = pd.to_datetime(
    requests_df["created_at"]
)

requests_df = requests_df.sort_values(
    "created_at",
    ascending=True,
)

if not feedback_df.empty:
    feedback_df["created_at"] = pd.to_datetime(
        feedback_df["created_at"]
    )

    feedback_df = feedback_df.sort_values(
        "created_at",
        ascending=False,
    )


def format_number(
    value: float | int | None,
    decimals: int = 0,
) -> str:
    """Format dashboard values safely when database columns are empty."""
    if value is None or pd.isna(value):
        return "N/A"

    if decimals == 0:
        return f"{value:,.0f}"

    return f"{value:,.{decimals}f}"


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

total_requests = len(requests_df)

avg_latency_ms = requests_df[
    "response_time_ms"
].mean()

avg_tokens = requests_df[
    "total_tokens"
].mean()

avg_rerank = requests_df[
    "top_rerank_score"
].mean()

avg_documents = requests_df[
    "retrieved_documents"
].mean()

avg_context_length = requests_df[
    "context_length"
].mean()

total_feedback = len(feedback_df)

helpful_rate: float | None = None

if total_feedback:
    helpful_rate = (
        feedback_df["helpful"].mean() * 100
    )


first_row = st.columns(4)

first_row[0].metric(
    "Requests",
    format_number(total_requests),
)

first_row[1].metric(
    "Avg Latency",
    (
        f"{format_number(avg_latency_ms)} ms"
        if not pd.isna(avg_latency_ms)
        else "N/A"
    ),
)

first_row[2].metric(
    "Avg Tokens",
    format_number(avg_tokens),
)

first_row[3].metric(
    "Avg Rerank",
    format_number(avg_rerank, decimals=3),
)


second_row = st.columns(4)

second_row[0].metric(
    "Avg Retrieved Docs",
    format_number(avg_documents, decimals=1),
)

second_row[1].metric(
    "Avg Context Length",
    format_number(avg_context_length),
)

second_row[2].metric(
    "Feedback Responses",
    format_number(total_feedback),
)

second_row[3].metric(
    "Helpful Rate",
    (
        f"{helpful_rate:.0f}%"
        if helpful_rate is not None
        else "N/A"
    ),
)

st.divider()


# ---------------------------------------------------------------------------
# Response time and tokens
# ---------------------------------------------------------------------------

left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Response Time")

    latency_chart = requests_df[
        [
            "created_at",
            "response_time_ms",
        ]
    ].set_index("created_at")

    st.line_chart(latency_chart)

with right_chart:
    st.subheader("Token Usage")

    token_chart = requests_df[
        [
            "created_at",
            "prompt_tokens",
            "completion_tokens",
        ]
    ].set_index("created_at")

    st.bar_chart(token_chart)

st.divider()


# ---------------------------------------------------------------------------
# Retrieval quality and feedback
# ---------------------------------------------------------------------------

left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Retrieval Quality")

    retrieval_chart = requests_df[
        [
            "created_at",
            "top_rerank_score",
            "top_vector_score",
        ]
    ].set_index("created_at")

    st.line_chart(retrieval_chart)

with right_chart:
    st.subheader("User Feedback")

    if feedback_df.empty:
        st.info("No feedback has been submitted yet.")
    else:
        feedback_summary = pd.DataFrame(
            {
                "Rating": [
                    "Helpful",
                    "Not Helpful",
                ],
                "Count": [
                    int(
                        feedback_df[
                            "helpful"
                        ].sum()
                    ),
                    int(
                        (
                            ~feedback_df[
                                "helpful"
                            ]
                        ).sum()
                    ),
                ],
            }
        )

        st.bar_chart(
            feedback_summary.set_index(
                "Rating"
            )
        )

st.divider()


# ---------------------------------------------------------------------------
# Recent requests
# ---------------------------------------------------------------------------

st.subheader("Recent Requests")

recent_requests = (
    requests_df.sort_values(
        "created_at",
        ascending=False,
    )
    .head(20)
    .copy()
)

recent_requests["created_at"] = (
    recent_requests["created_at"]
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)

recent_requests = recent_requests.rename(
    columns={
        "created_at": "Time",
        "question": "Question",
        "response_time_ms": "Latency (ms)",
        "retrieved_documents": "Documents",
        "total_tokens": "Tokens",
        "top_rerank_score": "Top Rerank",
    }
)

st.dataframe(
    recent_requests[
        [
            "Time",
            "Question",
            "Latency (ms)",
            "Documents",
            "Tokens",
            "Top Rerank",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ---------------------------------------------------------------------------
# Recent feedback
# ---------------------------------------------------------------------------

st.subheader("Recent Feedback")

if feedback_df.empty:
    st.info("No feedback has been submitted yet.")
else:
    recent_feedback = feedback_df.head(20).copy()

    recent_feedback["created_at"] = (
        recent_feedback["created_at"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    recent_feedback["helpful"] = (
        recent_feedback["helpful"].map(
            {
                True: "Helpful",
                False: "Not Helpful",
            }
        )
    )

    recent_feedback = recent_feedback.rename(
        columns={
            "created_at": "Time",
            "request_id": "Request ID",
            "helpful": "Rating",
            "comments": "Comments",
        }
    )

    st.dataframe(
        recent_feedback[
            [
                "Time",
                "Request ID",
                "Rating",
                "Comments",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )