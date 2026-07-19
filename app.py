"""
Clearvion HITL Interface
A Human-in-the-Loop validation tool for reviewing and adjusting
NLP-derived feature prioritisation rankings.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

st.set_page_config(page_title="Clearvion", layout="wide", page_icon="🎯")

DATA_DIR = "data"


# ----------------------------- Data loading -----------------------------

@st.cache_data
def load_data():
    cps = pd.read_csv(f"{DATA_DIR}/clearvion_cps_results.csv")
    reviews = pd.read_csv(f"{DATA_DIR}/clearvion_processed_reviews.csv")
    return cps, reviews


cps_df, reviews_df = load_data()


def recompute_cps(df, w_freq, w_sent, w_rel):
    total = w_freq + w_sent + w_rel
    if total == 0:
        w_freq, w_sent, w_rel = 1, 1, 1
        total = 3
    w_freq, w_sent, w_rel = w_freq / total, w_sent / total, w_rel / total
    df = df.copy()
    df["CPS_adjusted"] = (
        w_freq * df["Frequency_Score"]
        + w_sent * df["Sentiment_Score"]
        + w_rel * df["Relevance_Score"]
    )
    return df.sort_values("CPS_adjusted", ascending=False).reset_index(drop=True)


# ----------------------------- Session state -----------------------------

if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""
if "manual_ranks" not in st.session_state:
    st.session_state.manual_ranks = {}
if "comments" not in st.session_state:
    st.session_state.comments = {}
if "sus_answers" not in st.session_state:
    st.session_state.sus_answers = {}
if "qual_feedback" not in st.session_state:
    st.session_state.qual_feedback = {}


# ----------------------------- Sidebar: setup -----------------------------

st.sidebar.title("Clearvion")
st.sidebar.caption("Human-in-the-Loop feature prioritisation")

st.session_state.participant_id = st.sidebar.text_input(
    "Participant ID or name",
    value=st.session_state.participant_id,
    help="Used only to label your exported results file. Not stored anywhere else.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Scoring weights")
st.sidebar.caption(
    "The system defaults to equal weighting across frequency, sentiment, "
    "and relevance. Adjust the sliders to see how the ranking changes."
)
w_freq = st.sidebar.slider("Frequency weight", 0.0, 1.0, 0.33, 0.01)
w_sent = st.sidebar.slider("Sentiment weight", 0.0, 1.0, 0.33, 0.01)
w_rel = st.sidebar.slider("Relevance weight", 0.0, 1.0, 0.34, 0.01)

ranked_df = recompute_cps(cps_df, w_freq, w_sent, w_rel)

st.sidebar.markdown("---")
product_filter = st.sidebar.multiselect(
    "Filter evidence by product",
    options=sorted(reviews_df["productName"].dropna().unique().tolist()),
    default=sorted(reviews_df["productName"].dropna().unique().tolist()),
)


# ----------------------------- Main tabs -----------------------------

tab1, tab2, tab3 = st.tabs(
    ["1. Review & Adjust Rankings", "2. Usability Survey", "3. Submit Results"]
)

# ============================================================
# TAB 1: Ranking review
# ============================================================
with tab1:
    st.header("Feature Priority Rankings")
    st.markdown(
        "These rankings were produced automatically from customer review data "
        "using topic modelling and sentiment analysis. Your task is to review "
        "them, and use the **your rank** column to say where you think each "
        "feature should actually sit. If you disagree with the system, please "
        "add a short reason in the comment box below the table."
    )

    display_df = ranked_df.copy()
    display_df.insert(0, "System Rank", range(1, len(display_df) + 1))

    st.dataframe(
        display_df[
            [
                "System Rank",
                "Feature_Label",
                "Review_Count",
                "Frequency_Score",
                "Sentiment_Score",
                "Relevance_Score",
                "CPS_adjusted",
            ]
        ].rename(columns={"CPS_adjusted": "CPS"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Inspect a feature and set your own rank")

    feature_choice = st.selectbox(
        "Choose a feature to inspect",
        options=display_df["Feature_Label"].tolist(),
    )

    row = display_df[display_df["Feature_Label"] == feature_choice].iloc[0]
    topic_id = row["Topic"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("System rank", int(row["System Rank"]))
        st.metric("CPS score", round(row["CPS_adjusted"], 3))
        st.metric("Review count", int(row["Review_Count"]))

        max_rank = len(display_df)
        your_rank = st.number_input(
            "Your rank for this feature",
            min_value=1,
            max_value=max_rank,
            value=st.session_state.manual_ranks.get(
                feature_choice, int(row["System Rank"])
            ),
            key=f"rank_{feature_choice}",
        )
        st.session_state.manual_ranks[feature_choice] = your_rank

        comment = st.text_area(
            "Why? (optional, but helpful if your rank differs from the system's)",
            value=st.session_state.comments.get(feature_choice, ""),
            key=f"comment_{feature_choice}",
            height=100,
        )
        st.session_state.comments[feature_choice] = comment

    with col2:
        st.markdown(f"**Topic keywords:** {feature_choice}")
        matching_reviews = reviews_df[
            (reviews_df["topic"] == topic_id)
            & (reviews_df["productName"].isin(product_filter))
        ]
        st.caption(f"{len(matching_reviews)} reviews behind this feature (filtered)")

        sample = matching_reviews.sample(min(5, len(matching_reviews)), random_state=1) if len(matching_reviews) else matching_reviews
        for _, r in sample.iterrows():
            sentiment_color = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "⚪",
            }.get(str(r["sentiment"]).lower(), "⚪")
            with st.container(border=True):
                st.markdown(
                    f"{sentiment_color} **{r['productName']}** · "
                    f"{r['starRating']}★ · {str(r['sentiment']).title()}"
                )
                text = str(r["reviewText"])
                st.write(text[:400] + ("..." if len(text) > 400 else ""))

    st.markdown("---")
    st.subheader("Your final ranking order")
    manual_df = display_df.copy()
    manual_df["Your Rank"] = manual_df["Feature_Label"].map(
        lambda f: st.session_state.manual_ranks.get(f, None)
    )
    manual_df["Your Comment"] = manual_df["Feature_Label"].map(
        lambda f: st.session_state.comments.get(f, "")
    )
    st.dataframe(
        manual_df[["System Rank", "Feature_Label", "Your Rank", "Your Comment"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Tip: work through each feature in the inspector above to fill in "
        "'Your Rank' for all of them before moving to the next tab."
    )


# ============================================================
# TAB 2: SUS + qualitative feedback
# ============================================================
with tab2:
    st.header("Usability Survey")
    st.markdown(
        "Please rate how much you agree with each statement, based on your "
        "experience using this interface just now. This is the standard "
        "System Usability Scale (SUS)."
    )

    sus_items = [
        "I think that I would like to use this system frequently.",
        "I found the system unnecessarily complex.",
        "I thought the system was easy to use.",
        "I think that I would need the support of a technical person to use this system.",
        "I found the various functions in this system were well integrated.",
        "I thought there was too much inconsistency in this system.",
        "I would imagine that most people would learn to use this system very quickly.",
        "I found the system very cumbersome to use.",
        "I felt very confident using the system.",
        "I needed to learn a lot of things before I could get going with this system.",
    ]

    scale_labels = [
        "1 – Strongly disagree",
        "2",
        "3",
        "4",
        "5 – Strongly agree",
    ]

    for i, item in enumerate(sus_items):
        st.session_state.sus_answers[i] = st.radio(
            f"{i + 1}. {item}",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: scale_labels[x - 1],
            horizontal=True,
            key=f"sus_{i}",
            index=None,
        )

    st.markdown("---")
    st.subheader("A few open questions")
    st.session_state.qual_feedback["clarity"] = st.text_area(
        "Was the reasoning behind each ranking clear to you? Why or why not?",
        value=st.session_state.qual_feedback.get("clarity", ""),
    )
    st.session_state.qual_feedback["evidence"] = st.text_area(
        "Was the supporting review evidence useful when deciding whether to "
        "agree or disagree with a ranking?",
        value=st.session_state.qual_feedback.get("evidence", ""),
    )
    st.session_state.qual_feedback["confidence"] = st.text_area(
        "How confident are you in the final rankings you produced, and why?",
        value=st.session_state.qual_feedback.get("confidence", ""),
    )


def compute_sus_score(answers):
    if any(v is None for v in answers.values()) or len(answers) < 10:
        return None
    total = 0
    for i in range(10):
        v = answers[i]
        # Odd-numbered items (1-indexed): score = value - 1
        # Even-numbered items (1-indexed): score = 5 - value
        if (i + 1) % 2 == 1:
            total += v - 1
        else:
            total += 5 - v
    return total * 2.5


# ============================================================
# TAB 3: Submit / export
# ============================================================
with tab3:
    st.header("Submit Your Results")
    st.markdown(
        "This will package your rankings, comments, and survey answers into "
        "a single file. Please download it and send it back to the "
        "researcher — nothing is uploaded automatically."
    )

    sus_score = compute_sus_score(st.session_state.sus_answers)
    if sus_score is not None:
        st.success(f"Your SUS score: {sus_score:.1f} / 100")
    else:
        st.warning("Complete all 10 usability questions in Tab 2 to compute your SUS score.")

    missing_ranks = [
        f for f in ranked_df["Feature_Label"]
        if f not in st.session_state.manual_ranks
    ]
    if missing_ranks:
        st.info(
            f"You haven't set 'Your Rank' for {len(missing_ranks)} feature(s) yet. "
            "You can still submit, but consider going back to Tab 1 first."
        )

    if st.button("Prepare download", type="primary"):
        if not st.session_state.participant_id.strip():
            st.error("Please enter a participant ID or name in the sidebar first.")
        else:
            out = ranked_df.copy()
            out.insert(0, "System_Rank", range(1, len(out) + 1))
            out["Your_Rank"] = out["Feature_Label"].map(
                lambda f: st.session_state.manual_ranks.get(f, "")
            )
            out["Your_Comment"] = out["Feature_Label"].map(
                lambda f: st.session_state.comments.get(f, "")
            )
            out["Participant_ID"] = st.session_state.participant_id
            out["Weight_Frequency"] = w_freq
            out["Weight_Sentiment"] = w_sent
            out["Weight_Relevance"] = w_rel
            out["SUS_Score"] = sus_score if sus_score is not None else ""
            out["Feedback_Clarity"] = st.session_state.qual_feedback.get("clarity", "")
            out["Feedback_Evidence"] = st.session_state.qual_feedback.get("evidence", "")
            out["Feedback_Confidence"] = st.session_state.qual_feedback.get("confidence", "")
            out["Timestamp"] = datetime.now().isoformat()

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            fname = f"clearvion_results_{st.session_state.participant_id.strip().replace(' ', '_')}.csv"

            st.download_button(
                "Download my results (CSV)",
                data=csv_bytes,
                file_name=fname,
                mime="text/csv",
            )
            st.success("File ready. Click the button above to download, then send it to the researcher.")
