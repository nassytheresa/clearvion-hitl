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

# Human-readable names for each BERTopic topic, written from its top keywords.
# The raw keyword string (Feature_Label in the CSV) is still shown alongside
# these for transparency, but this is what participants see as the headline.
TOPIC_FRIENDLY_NAMES = {
    0: "Asana: Core Task & Project Management",
    1: "Jira: Issue Tracking & Team Workflows",
    2: "Day-to-Day Ease of Use for Task Tracking",
    3: "General Project Management Capability",
    4: "Trello: Task & Card Usability",
    5: "Trello: Board-Based Project Management",
    6: "Simplicity & Understandability of Boards",
    7: "Trello: Overall Ease of Use",
    8: "Trello: Visual Board Layout",
    9: "Kanban Board Functionality",
    10: "Trello: Team Task Clarity",
    11: "Positive Sentiment on Boards & Projects",
    12: "Trello: Simple, Organized Interface",
    13: "Trello: Card-Based Ease of Use",
    14: "Trello for Professional / Business Use",
    15: "Trello: Simplicity vs. Complex Needs",
    16: "Ticket Tracking & Enhancement Requests",
    17: "Card Due Dates & Lead Tracking",
    18: "Team Tracking Help (Mixed Clarity)",
    19: "Trello: Posting & Card Functionality",
    20: "Trello: Task Ordering & Organization",
}


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
    "Your name or ID (please fill this in first)",
    value=st.session_state.participant_id,
    help="Used only to label your exported results file. Not stored anywhere else.",
)
if not st.session_state.participant_id.strip():
    st.sidebar.warning("Please enter your name or ID above before starting.")

st.sidebar.markdown("---")
with st.sidebar.expander("Advanced (optional): explore scoring weights"):
    st.caption(
        "Not part of the task. Shows how the ranking would change under "
        "different weightings. Safe to ignore."
    )
    w_freq = st.slider("Frequency weight", 0.0, 1.0, 0.33, 0.01)
    w_sent = st.slider("Sentiment weight", 0.0, 1.0, 0.33, 0.01)
    w_rel = st.slider("Relevance weight", 0.0, 1.0, 0.34, 0.01)

ranked_df = recompute_cps(cps_df, w_freq, w_sent, w_rel)

with st.sidebar.expander("Advanced (optional): filter evidence by product"):
    product_filter = st.multiselect(
        "Product",
        options=sorted(reviews_df["productName"].dropna().unique().tolist()),
        default=sorted(reviews_df["productName"].dropna().unique().tolist()),
        label_visibility="collapsed",
    )


# ----------------------------- Welcome panel -----------------------------

with st.container(border=True):
    st.markdown("### 👋 Welcome")
    st.markdown(
        "**What this study is about:** Clearvion is a research tool built for "
        "a Master's thesis. It reads customer reviews for three project "
        "management tools, **Jira, Asana, and Trello**, and tries to work "
        "out which product features customers care about most, so a product "
        "team could use it to help decide what to prioritise next."
    )
    st.markdown(
        "**How it works, in short:** the tool takes 738 real customer "
        "reviews from G2 (a review site), and uses natural language "
        "processing to do three things. First, it groups similar feedback "
        "into 21 topics, things like \"Kanban boards\" or \"Trello's ease of "
        "use.\" Second, it works out whether customers feel positively or "
        "negatively about each topic. Third, it combines that into a single "
        "priority score for each one. To keep the task short, you'll review "
        "the top 10 out of the 21 by priority score, one at a time."
    )
    st.markdown(
        "**Why we need you:** an algorithm can spot patterns in review text, "
        "but it doesn't have real product judgement. Your role is to look at "
        "each topic and its supporting reviews, and tell us whether the "
        "system's ranking matches what you'd actually prioritise as a "
        "product manager. That comparison is the actual point of this study."
    )
    st.markdown(
        "**What you'll do (about 15 minutes):**\n"
        "1. Go through the top 10 features one at a time, and say yes or no on "
        "whether each ranking looks right to you.\n"
        "2. Answer a short survey about how easy this tool was to use. "
        "This part is about the tool, not the product features themselves.\n"
        "3. Download one results file and send it back."
    )
    with st.expander("Curious about the details? What is a 'CPS score'?"):
        st.markdown(
            "**CPS stands for Customer Priority Score.** It's a single number "
            "from **0 to 1** calculated for each of the 21 topics, meant "
            "to estimate how much attention that topic deserves. A higher "
            "number means higher priority.\n\n"
            "It's built from three ingredients, each also scored 0 to 1:\n"
            "- **Frequency**: how often customers mention this in reviews\n"
            "- **Sentiment**: how positively or negatively they talk about it\n"
            "- **Relevance**: how central this topic is within its reviews\n\n"
            "Right now these three are weighted equally. Your job isn't to "
            "check the maths. It's to use your own product judgement and "
            "say whether the resulting rank feels right, based on the "
            "evidence you're shown."
        )

n_total = len(cps_df)
n_ranked = len(st.session_state.manual_ranks)
st.progress(
    n_ranked / n_total if n_total else 0,
    text=f"Progress: {n_ranked} of {n_total} features ranked",
)

# ----------------------------- Step 1: Feature review -----------------------------

st.markdown("---")
st.header("Step 1: Review the Top 10 Features")
st.markdown(
    "The system found 21 topics in total, but to keep this task short, "
    "you'll only review the **top 10 by priority score**. These are the "
    "ones most relevant to an actual prioritisation decision. For each "
    "one below, see the reviews behind it, then say whether the ranking "
    "looks right to you."
)

display_df = ranked_df.copy()
display_df.insert(0, "System Rank", range(1, len(display_df) + 1))
TASK_SIZE = 10
display_df = display_df.head(TASK_SIZE)
n_total_features = len(display_df)


def stratified_sample(df, n=4, seed=1):
    """Pick a mix across sentiments instead of pure random, so a
    positive-skewed corpus doesn't hide the negative/neutral reviews
    that exist for this topic."""
    if len(df) == 0:
        return df
    groups = [g for _, g in df.groupby("sentiment") if len(g) > 0]
    picks = []
    per_group = max(1, n // max(1, len(groups)))
    for g in groups:
        picks.append(g.sample(min(per_group, len(g)), random_state=seed))
    result = pd.concat(picks)
    if len(result) < n:
        remaining = df.drop(result.index)
        extra = remaining.sample(min(n - len(result), len(remaining)), random_state=seed)
        result = pd.concat([result, extra])
    return result.sample(min(n, len(result)), random_state=seed)


for _, row in display_df.iterrows():
    feature_choice = row["Feature_Label"]
    topic_id = row["Topic"]
    friendly_name = TOPIC_FRIENDLY_NAMES.get(int(topic_id), feature_choice)

    st.markdown("---")
    st.subheader(f"#{int(row['System Rank'])}: {friendly_name}")
    st.caption(
        f"Raw model keywords: *{feature_choice}*. These are the words the "
        f"topic-modelling algorithm found most common in this group of "
        f"reviews; the name above is a plain-language summary of them."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write(
            f"**CPS score: {row['CPS_adjusted']:.2f}** out of 1.00 "
            f"(higher = higher priority)"
        )
        st.caption(
            f"Made up of: Frequency {row['Frequency_Score']:.2f}, "
            f"Sentiment {row['Sentiment_Score']:.2f}, "
            f"Relevance {row['Relevance_Score']:.2f}. Each is also out of 1.00"
        )
        st.write(f"Based on **{int(row['Review_Count'])} reviews**")

        already_disagree = feature_choice in st.session_state.manual_ranks and (
            st.session_state.manual_ranks[feature_choice] != int(row["System Rank"])
        )
        agree = st.radio(
            "Does this ranking look right to you?",
            options=["Yes, looks right", "No, I'd rank it differently"],
            index=1 if already_disagree else 0,
            key=f"agree_{feature_choice}",
        )

        if agree == "No, I'd rank it differently":
            your_rank = st.number_input(
                "What rank should it be? (1 = highest priority)",
                min_value=1,
                max_value=n_total_features,
                value=st.session_state.manual_ranks.get(
                    feature_choice, int(row["System Rank"])
                ),
                key=f"rank_{feature_choice}",
            )
            st.session_state.manual_ranks[feature_choice] = your_rank
            comment = st.text_area(
                "Why? (optional)",
                value=st.session_state.comments.get(feature_choice, ""),
                key=f"comment_{feature_choice}",
                height=150,
            )
            st.session_state.comments[feature_choice] = comment
        else:
            st.session_state.manual_ranks[feature_choice] = int(row["System Rank"])
            st.session_state.comments[feature_choice] = ""

    with col2:
        st.markdown("**What customers are saying about this:**")
        matching_reviews = reviews_df[
            (reviews_df["topic"] == topic_id)
            & (reviews_df["productName"].isin(product_filter))
        ]

        sample = stratified_sample(matching_reviews, n=4)
        sentiment_counts_here = matching_reviews["sentiment"].value_counts()
        if len(sentiment_counts_here) == 1:
            only_sentiment = sentiment_counts_here.index[0]
            st.caption(
                f"All {len(matching_reviews)} reviews found for this topic were "
                f"classified {str(only_sentiment).lower()}. That's the actual "
                f"data, not a display choice."
            )
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
                st.write(str(r["reviewText"]))

st.markdown("---")
n_done = len(st.session_state.manual_ranks)
st.progress(n_done / n_total_features, text=f"{n_done} of {n_total_features} features reviewed")
if n_done < n_total_features:
    st.info("Scroll up to review any remaining features before moving to Step 2.")
else:
    st.success("All features reviewed. Continue to Step 2 below.")

with st.expander("See the full list with your answers so far"):
    manual_df = display_df.copy()
    manual_df["Feature Name"] = manual_df["Topic"].map(
        lambda t: TOPIC_FRIENDLY_NAMES.get(int(t), "")
    )
    manual_df["Your Rank"] = manual_df["Feature_Label"].map(
        lambda f: st.session_state.manual_ranks.get(f, None)
    )
    manual_df["Your Comment"] = manual_df["Feature_Label"].map(
        lambda f: st.session_state.comments.get(f, "")
    )
    st.dataframe(
        manual_df[["System Rank", "Feature Name", "Feature_Label", "Your Rank", "Your Comment"]]
        .rename(columns={"Feature_Label": "Raw Keywords"}),
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------- Step 2: Survey -----------------------------

st.markdown("---")
st.header("Step 2: Usability Survey")
st.markdown(
    "These questions are about **your experience using this tool**, not "
    "about whether you agreed with the feature rankings. There are no "
    "right or wrong answers here. This is just to check whether the "
    "tool itself is clear and easy to use for a product manager."
)
st.info(
    "For each statement below, pick how much you agree, from "
    "**1 (strongly disagree)** to **5 (strongly agree)**. This is a "
    "standard, widely-used usability questionnaire called the SUS "
    "(System Usability Scale). You don't need to know that to answer it, "
    "just go with your gut reaction."
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


# ----------------------------- Step 3: Submit / export -----------------------------

st.markdown("---")
st.header("Step 3: Submit Your Results")
st.markdown(
    "This will package your rankings, comments, and survey answers into "
    "a single file. Please download it and send it back to the "
    "researcher. Nothing is uploaded automatically."
)

sus_score = compute_sus_score(st.session_state.sus_answers)
if sus_score is not None:
    st.success(f"Your SUS score: {sus_score:.1f} / 100")
    if sus_score >= 68:
        st.caption(
            "For context: 68 is considered an average, acceptable score for "
            "this kind of questionnaire. Your score is at or above that."
        )
    else:
        st.caption(
            "For context: 68 is considered an average, acceptable score for "
            "this kind of questionnaire. Your score is below that, which is "
            "useful feedback, not a problem with your answers."
        )
else:
    st.warning("Complete all 10 usability questions in Step 2 to compute your SUS score.")

missing_ranks = [
    f for f in display_df["Feature_Label"]
    if f not in st.session_state.manual_ranks
]
if missing_ranks:
    st.info(
        f"You haven't reviewed {len(missing_ranks)} feature(s) yet. "
        "You can still submit, but consider scrolling up to Step 1 first."
    )

if st.button("Prepare download", type="primary"):
    if not st.session_state.participant_id.strip():
        st.error("Please enter a participant ID or name in the sidebar first.")
    else:
        out = display_df.copy()
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
