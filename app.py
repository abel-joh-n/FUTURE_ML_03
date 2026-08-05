import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer
)
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Screening & Candidate Ranking System")

st.markdown(
"""
This application compares resumes with a Job Description using **Natural Language Processing (NLP)** and **Machine Learning**.

### Features
- Resume preprocessing
- Skill extraction
- TF-IDF Vectorization
- Cosine Similarity
- Candidate Ranking
- Missing Skill Detection
"""
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Resume.csv")

df = load_data()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

stop_words = ENGLISH_STOP_WORDS

def clean_resume(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["Cleaned_Resume"] = df["Resume_str"].apply(clean_resume)

# --------------------------------------------------
# SKILL DATABASE
# --------------------------------------------------

skills = [

    # Programming

    "python","java","c","c++","c#","javascript",
    "typescript","php","go","swift","kotlin",

    # Web

    "html","css","bootstrap","react","angular",
    "vue","node","express","django","flask",
    "fastapi",

    # Databases

    "sql","mysql","postgresql","mongodb",
    "sqlite","oracle","firebase",

    # Data Science

    "numpy","pandas","matplotlib","seaborn",
    "plotly","scikit-learn","tensorflow",
    "keras","pytorch","opencv",
    "machine learning",
    "deep learning",
    "artificial intelligence",

    # Cloud

    "aws","azure","gcp",
    "docker","kubernetes",
    "git","github",

    # BI

    "power bi","tableau","excel",

    # Soft Skills

    "communication",
    "leadership",
    "teamwork",
    "problem solving",
    "critical thinking",
    "time management",
    "project management",
    "customer service"
]

# --------------------------------------------------
# SKILL EXTRACTION
# --------------------------------------------------

def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills:

        if skill in text:
            found.append(skill)

    return sorted(list(set(found)))

df["Extracted Skills"] = df["Cleaned_Resume"].apply(extract_skills)

# --------------------------------------------------
# JOB DESCRIPTION INPUT
# --------------------------------------------------

st.header("📝 Job Description")

job_description = st.text_area(

    "Paste the Job Description",

    height=250,

    placeholder="""
Example

Machine Learning Engineer

Required Skills

Python
Machine Learning
Deep Learning
TensorFlow
Scikit-learn
Pandas
NumPy
SQL
Git
Docker
AWS
Communication
Problem Solving
"""
)

analyze = st.button(
    "🔍 Analyze Candidates",
    use_container_width=True
)

# --------------------------------------------------
# ANALYZE RESUMES
# --------------------------------------------------

if analyze:

    if job_description.strip() == "":
        st.warning("Please enter a Job Description.")
        st.stop()

    # Clean Job Description
    cleaned_job = clean_resume(job_description)

    # Extract Required Skills
    job_skills = extract_skills(cleaned_job)

    # TF-IDF Similarity
    documents = [cleaned_job] + df["Cleaned_Resume"].tolist()

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )

    df["Match Score"] = similarity.flatten() * 100

    # Missing Skills
    df["Missing Skills"] = df["Extracted Skills"].apply(
        lambda x: sorted(list(set(job_skills) - set(x)))
    )

    # Ranking
    top_candidates = df.sort_values(
        by="Match Score",
        ascending=False
    )

    # --------------------------------------------
    # KPI CARDS
    # --------------------------------------------

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🏆 Highest Match",
        f"{top_candidates.iloc[0]['Match Score']:.2f}%"
    )

    c2.metric(
        "📊 Average Match",
        f"{df['Match Score'].mean():.2f}%"
    )

    c3.metric(
        "👥 Candidates",
        len(df)
    )

    # --------------------------------------------
    # BAR CHART
    # --------------------------------------------

    st.subheader("📈 Top 10 Candidate Scores")

    chart = (
        top_candidates.head(10)
        .set_index("Category")["Match Score"]
    )

    st.bar_chart(chart)

    # --------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------

    st.subheader("🏆 Top 10 Candidates")

    display = top_candidates[
        [
            "Category",
            "Match Score",
            "Extracted Skills",
            "Missing Skills"
        ]
    ].head(10)

    st.dataframe(
        display,
        use_container_width=True
    )

    # --------------------------------------------
    # BEST CANDIDATE
    # --------------------------------------------

    st.subheader("⭐ Best Candidate")

    best = top_candidates.iloc[0]

    st.success(
        f"""
Highest Match Score

{best['Match Score']:.2f}%
"""
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### ✅ Skills Found")

        if len(best["Extracted Skills"]) == 0:
            st.write("No matching skills detected.")
        else:
            for skill in best["Extracted Skills"]:
                st.write("•", skill)

    with right:

        st.markdown("### ❌ Missing Skills")

        if len(best["Missing Skills"]) == 0:
            st.write("No missing skills.")
        else:
            for skill in best["Missing Skills"]:
                st.write("•", skill)

    # --------------------------------------------
    # SEARCH CANDIDATES
    # --------------------------------------------

    st.subheader("🔎 Search Candidates")

    search = st.text_input(
        "Search by Resume Category",
        placeholder="Example: INFORMATION-TECHNOLOGY"
    )

    if search:

        filtered = top_candidates[
            top_candidates["Category"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            filtered[
                [
                    "Category",
                    "Match Score",
                    "Extracted Skills",
                    "Missing Skills"
                ]
            ],
            use_container_width=True
        )

    # --------------------------------------------
    # CATEGORY DISTRIBUTION
    # --------------------------------------------

    st.subheader("📊 Resume Category Distribution")

    category_count = (
        df["Category"]
        .value_counts()
    )

    st.bar_chart(category_count)

    # --------------------------------------------
    # MATCH SCORE DISTRIBUTION
    # --------------------------------------------

    st.subheader("📈 Match Score Distribution")

    st.subheader("📈 Match Score Distribution")

    histogram = (
        pd.cut(df["Match Score"], bins=10)
        .value_counts()
        .sort_index()
    )

    hist_df = histogram.reset_index()
    hist_df.columns = ["Match Range", "Candidates"]

    hist_df["Match Range"] = hist_df["Match Range"].astype(str)

    st.bar_chart(
        hist_df.set_index("Match Range")
    )

    # --------------------------------------------
    # RESUME PREVIEW
    # --------------------------------------------

    st.subheader("📈 Top 20 Match Scores")

    chart = top_candidates.head(20)

    st.bar_chart(
        chart.set_index("Category")["Match Score"]
    )
    # --------------------------------------------
    # DOWNLOAD CSV
    # --------------------------------------------

    csv = top_candidates.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Ranking Results",
        data=csv,
        file_name="Resume_Ranking_Results.csv",
        mime="text/csv",
        use_container_width=True
    )

    # --------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------

    st.subheader("📋 Screening Summary")

    st.info(f"""
Total Resumes : {len(df)}

Highest Match : {top_candidates.iloc[0]['Match Score']:.2f} %

Average Match : {df['Match Score'].mean():.2f} %

Top Category : {top_candidates.iloc[0]['Category']}
""")

    # --------------------------------------------
    # FOOTER
    # --------------------------------------------

    st.divider()

    st.markdown(
        """
---
### 📄 Resume Screening & Candidate Ranking System

Built using

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- NLP Text Processing

Developed for **Future Interns Machine Learning Task 3**
"""
    )
