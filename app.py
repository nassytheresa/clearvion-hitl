"""
Clearvion HITL Interface
A simple Human-in-the-Loop tool: rank features yourself, then compare
your ranking against the system's, then reflect on why.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

st.set_page_config(page_title="Clearvion", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #F4F5FB 0%, #FFFFFF 250px);
    }
    .hero-banner {
        background: linear-gradient(120deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25);
    }
    .hero-banner h1 {
        color: white !important;
        font-size: 1.9rem;
        margin-bottom: 0.4rem;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0;
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: linear-gradient(120deg, #6366F1, #8B5CF6);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
    .feature-card {
        background: white;
        border: 1px solid #E9E9F5;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 2px 8px rgba(30, 27, 46, 0.04);
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #E9E9F5;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #6366F1, #8B5CF6);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.6rem;
    }
    div[data-testid="stMetric"] {
        background: #F4F5FB;
        border-radius: 12px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"

# The 10 features shown to participants. Chosen from all 21 topics found by
# BERTopic, filtered to those with at least 10 supporting reviews (so each
# one has enough real evidence behind it), then ranked by CPS score.
SELECTED_TOPICS = [0, 1, 8, 10, 7, 4, 2, 3, 9, 6]

# Descriptive name and one-line plain-English description for each, written
# from the topic's actual keywords and review content.
FEATURE_INFO = {
    0: {
        "name": "Everyday Task & Project Management",
        "description": "Creating, assigning, and tracking day-to-day tasks and projects.",
        "tool": "Asana",
    },
    1: {
        "name": "Issue Tracking Across Team Workflows",
        "description": "Logging, tracking, and moving issues through a team's workflow.",
        "tool": "Jira",
    },
    8: {
        "name": "Visual Board Layout & Design",
        "description": "How boards look and feel: colours, layout, visual organisation.",
        "tool": "Trello",
    },
    10: {
        "name": "Clarity of Tasks for Teams",
        "description": "Whether team members can easily tell what needs to be done.",
        "tool": "Trello",
    },
    7: {
        "name": "Overall Ease of Use & Learning Curve",
        "description": "How easy the tool is to pick up and use without training.",
        "tool": "Trello",
    },
    4: {
        "name": "Everyday Usability of Tasks & Cards",
        "description": "How smooth it feels to create, edit, and move tasks or cards.",
        "tool": "Trello",
    },
    2: {
        "name": "Day-to-Day Task Tracking Experience",
        "description": "The daily experience of logging and tracking ongoing tasks.",
        "tool": None,
    },
    3: {
        "name": "Broad Project Management Capability",
        "description": "General-purpose project planning and management features.",
        "tool": None,
    },
    9: {
        "name": "Kanban Board Functionality",
        "description": "Kanban-style boards for visualising work stages.",
        "tool": None,
    },
    6: {
        "name": "Simplicity of Board-Based Organisation",
        "description": "How intuitive board-based organisation feels to new users.",
        "tool": None,
    },
}


# ----------------------------- Data loading -----------------------------

@st.cache_data
def load_data():
    cps = pd.read_csv(f"{DATA_DIR}/clearvion_cps_results.csv")
    reviews = pd.read_csv(f"{DATA_DIR}/clearvion_processed_reviews.csv")
    return cps, reviews


cps_df, reviews_df = load_data()

features_df = cps_df[cps_df["Topic"].isin(SELECTED_TOPICS)].copy()
features_df = features_df.sort_values("CPS", ascending=False).reset_index(drop=True)
features_df.insert(0, "AI_Rank", range(1, len(features_df) + 1))
features_df["Name"] = features_df["Topic"].map(lambda t: FEATURE_INFO[int(t)]["name"])
features_df["Description"] = features_df["Topic"].map(lambda t: FEATURE_INFO[int(t)]["description"])
N_FEATURES = len(features_df)


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


# ----------------------------- Session state -----------------------------

if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""
if "your_ranks" not in st.session_state:
    st.session_state.your_ranks = {}
if "sus_answers" not in st.session_state:
    st.session_state.sus_answers = {}
if "reflection" not in st.session_state:
    st.session_state.reflection = {}


# ----------------------------- Sidebar -----------------------------

st.sidebar.title("Clearvion")
st.sidebar.caption("Feature prioritisation, human-in-the-loop")

st.session_state.participant_id = st.sidebar.text_input(
    "Your name or ID (please fill this in first)",
    value=st.session_state.participant_id,
    help="Used only to label your results. Not stored anywhere else.",
)
if not st.session_state.participant_id.strip():
    st.sidebar.warning("Please enter your name or ID above before starting.")

with st.sidebar.expander("Advanced (optional): filter evidence by product"):
    product_filter = st.multiselect(
        "Product",
        options=sorted(reviews_df["productName"].dropna().unique().tolist()),
        default=sorted(reviews_df["productName"].dropna().unique().tolist()),
        label_visibility="collapsed",
    )


# ----------------------------- Intro -----------------------------

st.markdown(
    """
    <div class="hero-banner">
    <h1>🎯 What this tool does</h1>
    <p><b>It helps decide which product features to build next</b>, by comparing your judgement against a system that ranked features based on real customer reviews.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("### How this works (about 10 minutes)")
    st.markdown(
        "1. **You rank 10 features yourself**, 1 to 10, based only on the "
        "name and description. No scores, no reviews, just your own "
        "judgement.\n"
        "2. **See the comparison** between your ranking and the system's, "
        "with the customer reviews behind the system's ranking available "
        "if you want to check them.\n"
        "3. **Tell us why** you ranked things the way you did.\n"
        "4. **A short usability survey**, then you're done."
    )
    st.caption(
        "These 10 features were picked from 21 the system found in 738 "
        "customer reviews, filtered to the ones with enough reviews behind "
        "them to be worth evaluating."
    )


# ----------------------------- Step 1: Rank the features -----------------------------

st.markdown("---")
st.markdown('<div class="step-badge">STEP 1</div>', unsafe_allow_html=True)
st.header("Rank These 10 Features")
st.info(
    "**What are these?** Each of the 10 items below represents a specific "
    "capability area that customers actually talk about in their reviews "
    "of Jira, Asana, and Trello, things like \"Kanban Board Functionality\" "
    "or \"Issue Tracking Across Team Workflows.\" They were found by "
    "running NLP on 738 real customer reviews.\n\n"
    "**How do I rank them?** Imagine you're the product manager for one "
    "of these tools, deciding what to focus on next. Based only on the "
    "name and description shown, which of these customer-driven themes "
    "would you personally consider most important?"
)
st.markdown(
    "Give each feature a rank from **1 (highest priority)** to "
    "**10 (lowest priority)**. Use each number once."
)

for _, row in features_df.iterrows():
    feature_key = row["Feature_Label"]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{row['Name']}**")
        st.caption(row["Description"])
    with col2:
        current = st.session_state.your_ranks.get(feature_key, None)
        val = st.number_input(
            "Your rank",
            min_value=1,
            max_value=N_FEATURES,
            value=current if current else 1,
            key=f"rank_{feature_key}",
            label_visibility="collapsed",
        )
        st.session_state.your_ranks[feature_key] = val

your_rank_values = list(st.session_state.your_ranks.values())
duplicates = len(your_rank_values) != len(set(your_rank_values))
all_filled = len(st.session_state.your_ranks) == N_FEATURES

if duplicates:
    st.error("You've used the same number more than once. Please make sure each rank from 1 to 10 is used exactly once.")
elif all_filled:
    st.success("All 10 features ranked, and no duplicate numbers. Continue to Step 2 below.")
else:
    st.info(f"{len(st.session_state.your_ranks)} of {N_FEATURES} features ranked so far.")


# ----------------------------- Step 2: Compare -----------------------------

st.markdown("---")
st.markdown('<div class="step-badge">STEP 2</div>', unsafe_allow_html=True)
st.header("Compare Your Ranking to the System's")

if duplicates or not all_filled:
    st.warning("Finish Step 1 with no duplicate numbers to see the comparison.")
else:
    compare_df = features_df.copy()
    compare_df["Your_Rank"] = compare_df["Feature_Label"].map(st.session_state.your_ranks)
    compare_df = compare_df.sort_values("Your_Rank")
    compare_df["Gap"] = (compare_df["Your_Rank"] - compare_df["AI_Rank"]).abs()

    def match_label(gap):
        if gap == 0:
            return "🟢 Exact match"
        elif gap <= 2:
            return "🟡 Close"
        else:
            return "🔴 Far apart"

    compare_df["Match"] = compare_df["Gap"].map(match_label)

    display_table = compare_df[["Your_Rank", "AI_Rank", "Name", "Match"]].rename(
        columns={"Your_Rank": "Your Rank", "AI_Rank": "System Rank", "Name": "Feature"}
    )
    st.dataframe(
        display_table.style.background_gradient(
            subset=["Your Rank", "System Rank"], cmap="Purples", vmin=1, vmax=N_FEATURES
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Want to see why the system ranked things this way?**")
    for _, row in compare_df.iterrows():
        topic_id = int(row["Topic"])
        with st.expander(f"{row['Name']}: system said #{int(row['AI_Rank'])}, you said #{int(row['Your_Rank'])}"):
            st.write(
                f"**CPS score: {row['CPS']:.2f}** out of 1.00, made up of "
                f"Frequency {row['Frequency_Score']:.2f}, Sentiment "
                f"{row['Sentiment_Score']:.2f}, Relevance {row['Relevance_Score']:.2f}."
            )
            matching_reviews = reviews_df[
                (reviews_df["topic"] == topic_id)
                & (reviews_df["productName"].isin(product_filter))
            ]
            sample = stratified_sample(matching_reviews, n=3)
            for _, r in sample.iterrows():
                sentiment_color = {
                    "positive": "🟢", "negative": "🔴", "neutral": "⚪",
                }.get(str(r["sentiment"]).lower(), "⚪")
                with st.container(border=True):
                    st.markdown(f"{sentiment_color} {r['starRating']}★ · {str(r['sentiment']).title()}")
                    st.write(str(r["reviewText"]))


# ----------------------------- Step 3: Reflection -----------------------------

st.markdown("---")
st.markdown('<div class="step-badge">STEP 3</div>', unsafe_allow_html=True)
st.header("Why Did You Rank Things This Way?")
st.session_state.reflection["reasoning"] = st.text_area(
    "In a few sentences, what did you base your ranking on? (e.g. your own "
    "experience with similar tools, what seemed intuitively important, etc.)",
    value=st.session_state.reflection.get("reasoning", ""),
)
st.session_state.reflection["biggest_difference"] = st.text_area(
    "Looking at the comparison in Step 2, what's the biggest difference "
    "between your ranking and the system's, and why do you think that "
    "happened?",
    value=st.session_state.reflection.get("biggest_difference", ""),
)


# ----------------------------- Step 4: Usability survey -----------------------------

st.markdown("---")
st.markdown('<div class="step-badge">STEP 4</div>', unsafe_allow_html=True)
st.header("Quick Usability Survey")
st.markdown(
    "These questions are about **your experience using this tool**, not "
    "about the features themselves. There are no right or wrong answers."
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
scale_labels = ["1 - Strongly disagree", "2", "3", "4", "5 - Strongly agree"]

for i, item in enumerate(sus_items):
    st.session_state.sus_answers[i] = st.radio(
        f"{i + 1}. {item}",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: scale_labels[x - 1],
        horizontal=True,
        key=f"sus_{i}",
        index=None,
    )


def compute_sus_score(answers):
    if any(v is None for v in answers.values()) or len(answers) < 10:
        return None
    total = 0
    for i in range(10):
        v = answers[i]
        if (i + 1) % 2 == 1:
            total += v - 1
        else:
            total += 5 - v
    return total * 2.5


# ----------------------------- Step 5: Submit -----------------------------

st.markdown("---")
st.markdown('<div class="step-badge">STEP 5</div>', unsafe_allow_html=True)
st.header("Submit")

sus_score = compute_sus_score(st.session_state.sus_answers)
if sus_score is not None:
    st.success(f"Your SUS score: {sus_score:.1f} / 100")
else:
    st.warning("Complete all 10 usability questions in Step 4 to finish.")


def try_send_email(csv_bytes, fname, participant_id):
    try:
        sender = st.secrets["email"]["sender"]
        app_password = st.secrets["email"]["app_password"]
        recipient = st.secrets["email"]["recipient"]
    except Exception:
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = f"Clearvion pilot results: {participant_id}"
        msg.attach(MIMEText(
            f"Pilot results from: {participant_id}\nSubmitted: {datetime.now().isoformat()}",
            "plain",
        ))
        part = MIMEBase("application", "octet-stream")
        part.set_payload(csv_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={fname}")
        msg.attach(part)
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, recipient, msg.as_string())
        return True
    except Exception:
        return False


if st.button("Submit my results", type="primary"):
    if not st.session_state.participant_id.strip():
        st.error("Please enter a participant ID or name in the sidebar first.")
    elif duplicates or not all_filled:
        st.error("Please finish Step 1 (rank all 10, no duplicates) before submitting.")
    else:
        out = features_df.copy()
        out["Your_Rank"] = out["Feature_Label"].map(st.session_state.your_ranks)
        out["Participant_ID"] = st.session_state.participant_id
        out["SUS_Score"] = sus_score if sus_score is not None else ""
        out["Reasoning"] = st.session_state.reflection.get("reasoning", "")
        out["Biggest_Difference"] = st.session_state.reflection.get("biggest_difference", "")
        out["Timestamp"] = datetime.now().isoformat()

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        fname = f"clearvion_results_{st.session_state.participant_id.strip().replace(' ', '_')}.csv"

        sent = try_send_email(csv_bytes, fname, st.session_state.participant_id)
        if sent:
            st.success("Done! Your results were sent directly to the researcher. Thank you.")
        else:
            st.warning("Automatic sending isn't set up yet. Please download and send this file to the researcher.")
            st.download_button("Download my results (CSV)", data=csv_bytes, file_name=fname, mime="text/csv")

        exact_matches = sum(
            1 for _, r in out.iterrows() if int(r["Your_Rank"]) == int(r["AI_Rank"])
        )
        close_matches = sum(
            1 for _, r in out.iterrows() if abs(int(r["Your_Rank"]) - int(r["AI_Rank"])) <= 2
        )
        biggest_gap_row = out.loc[
            (out["Your_Rank"] - out["AI_Rank"]).abs().idxmax()
        ]

        with st.container(border=True):
            st.markdown("### Here's a quick look at what you just did")
            st.write(
                f"You matched the system's rank exactly on **{exact_matches} of {N_FEATURES}** "
                f"features, and were within 2 places on **{close_matches} of {N_FEATURES}**."
            )
            st.write(
                f"Your biggest disagreement with the system was on "
                f"**{biggest_gap_row['Name']}**: you ranked it #{int(biggest_gap_row['Your_Rank'])}, "
                f"the system ranked it #{int(biggest_gap_row['AI_Rank'])}."
            )
            st.markdown(
                "**Why this matters:** the goal of this research is to test whether a tool "
                "like this can genuinely help product teams turn large volumes of customer "
                "feedback into decisions, without replacing the judgement a person like you "
                "brings. Your ranking, and where it agreed or disagreed with the system, is "
                "exactly the evidence needed to answer that. Whether you matched the system "
                "closely or not, both outcomes are useful. Thank you for your time."
            )
