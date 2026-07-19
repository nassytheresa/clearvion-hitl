# Clearvion HITL Interface

A pilot-ready Human-in-the-Loop interface for the Clearvion feature
prioritisation tool. Participants can review the system's rankings,
inspect the reviews behind each one, set their own rank, leave a
comment, complete a usability survey, and download a single results
file to send back to you.

## Files

- `app.py` — the Streamlit application
- `requirements.txt` — Python dependencies
- `data/clearvion_cps_results.csv` — the 21 ranked features from your NLP pipeline
- `data/clearvion_processed_reviews.csv` — the underlying reviews (used for evidence display)

## Run it locally (fastest way to check it works)

```bash
pip install -r requirements.txt
streamlit run app.py
```

It'll open in your browser at `http://localhost:8501`.

## Deploy it so participants can use it remotely (recommended)

1. Create a free account at https://streamlit.io/cloud if you don't have one.
2. Push this whole folder to a **public or private GitHub repo**
   (Streamlit Cloud needs GitHub access to deploy it).
   ```bash
   git init
   git add .
   git commit -m "Clearvion HITL pilot interface"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
3. On https://share.streamlit.io, click "New app", point it at your repo,
   and set the main file to `app.py`. Deploy.
4. You'll get a public URL like `https://your-app-name.streamlit.app`.
   Send that link to each participant.

## What participants do

1. Enter their name/ID in the sidebar.
2. **Tab 1** — review the system's rankings, inspect individual features
   (with real review evidence), and set their own rank + comment for each one.
3. **Tab 2** — complete the 10-item usability survey (SUS) and three
   open-text questions.
4. **Tab 3** — download a CSV with everything: their rankings, comments,
   SUS score, and qualitative answers. They email that file back to you.

## What you do with the results afterwards

Each downloaded CSV gives you, per participant:
- Their rank for every feature (compare to `System_Rank` for Precision@k / nDCG)
- Their written justification for any disagreement
- Their SUS score
- Their answers to the three open questions (for thematic analysis)

Collect all 3–5 participant files, then you're ready to compute:
- **Krippendorff's Alpha** across participants' "Your_Rank" columns, to check
  inter-rater agreement before treating them as a benchmark
- **Precision@k** and **nDCG** comparing `System_Rank` against each participant's rank
- **Mean SUS score** across participants, against your target of 68+

## Notes

- The sidebar weight sliders let a participant see how the ranking changes
  under different weightings. This isn't required for the study, but it's
  a nice way to demonstrate the CPS formula is transparent, not a black box.
- Nothing is stored on a server. Each participant's data only exists in
  their own browser session until they click download.
- Swap in an updated `clearvion_cps_results.csv` any time — the app doesn't
  need any code changes if the same column names are used.
