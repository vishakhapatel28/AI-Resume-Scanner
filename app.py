import streamlit as st
from utils import extract_text_from_pdf, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import tempfile
import os

st.title("AI Resume Scanner")

# 🔹 Toggle for method
method = st.radio("Choose Matching Method", ["TF-IDF", "BERT"])

uploaded_files = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=True)

job_description = st.text_area("Paste the job description")

if st.button("Analyze Resumes"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not job_description:
        st.error("Please enter a job description.")
    else:

        # ---------------- SKILL LIST ----------------
        skills = [
            "python","sql","machine learning","deep learning","nlp",
            "pandas","numpy","tensorflow","scikit-learn",
            "tableau","power bi","data analysis","statistics"
        ]

        def extract_skills(text):
            text = text.lower()
            return [skill for skill in skills if skill in text]

        resumes = []
        names = []

        # -------- Extract resume text --------
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            text = extract_text_from_pdf(tmp_path)
            processed = preprocess_text(text)

            resumes.append(processed)
            names.append(file.name)

            os.unlink(tmp_path)

        # -------- Process job description --------
        jd_processed = preprocess_text(job_description)
        jd_skills = extract_skills(job_description)

        # -------- Similarity Calculation --------
        if method == "TF-IDF":

            all_texts = resumes + [jd_processed]

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            jd_vector = tfidf_matrix[-1]
            resume_vectors = tfidf_matrix[:-1]

            similarities = cosine_similarity(resume_vectors, jd_vector).flatten()
            scores = similarities * 100

        else:

            model = SentenceTransformer('all-MiniLM-L6-v2')

            resume_embeddings = model.encode(resumes)
            jd_embedding = model.encode([jd_processed])

            similarities = cosine_similarity(resume_embeddings, jd_embedding).flatten()
            scores = similarities * 100

        # -------- Skill Matching --------
        skill_matches = []
        skill_missing = []

        for resume_text in resumes:
            rskills = extract_skills(resume_text)

            matched = list(set(rskills) & set(jd_skills))
            missing = list(set(jd_skills) - set(rskills))

            skill_matches.append(", ".join(matched))
            skill_missing.append(", ".join(missing))

        # -------- Create Results Table --------
        results = pd.DataFrame({
            "Resume Name": names,
            "Match Score (%)": scores,
            "Matched Skills": skill_matches,
            "Missing Skills": skill_missing
        })

        results = results.sort_values(by="Match Score (%)", ascending=False)

        # -------- Dashboard Stats --------
        st.subheader("Resume Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Resumes", len(results))
        col2.metric("Top Score", f"{results['Match Score (%)'].max():.2f}%")
        col3.metric("Average Score", f"{results['Match Score (%)'].mean():.2f}%")

        # -------- Show Table --------
        st.table(results)

        # -------- Chart --------
        st.bar_chart(results.set_index("Resume Name")["Match Score (%)"])

        # -------- Download Button --------
        csv = results.to_csv(index=False)

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="resume_ranking_results.csv",
            mime="text/csv"
        )