import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from transformers import pipeline
from resumescreen import evaluate_resume # Your backend function

# Load AI model (Zero-shot classifier from HuggingFace)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Extract text from resumes
def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# Streamlit UI
st.title("🤖 AI-Powered Resume Screening")

# Collect details
st.subheader("🔹 Candidate Information")
name = st.text_input("👤 Full Name")
email = st.text_input("📧 Email")
phone = st.text_input("📱 Contact Number")

# Upload Resume
st.subheader("📂 Upload Resume")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    if st.button("🚀 Evaluate with AI"):
        with st.spinner("Analyzing resume..."):
            # Save temp file
            os.makedirs("temp_resume", exist_ok=True)
            file_path = os.path.join("temp_resume", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text
            resume_text = extract_text(file_path)

            # AI-based skill evaluation
            candidate_labels = ["Python", "Machine Learning", "Data Science", "Communication", "Leadership"]
            ai_result = classifier(resume_text, candidate_labels)

            # Backend evaluation
            backend_result = evaluate_resume(file_path)

            # Display Results
            st.subheader("📊 AI Insights")
            for label, score in zip(ai_result["labels"], ai_result["scores"]):
                emoji = "🙂" if score > 0.5 else "sorry"
                st.write(f"{emoji} {label}: {round(score*100,2)}%")

            st.subheader("🛠 Backend Evaluation")
            for key, value in backend_result.items():
                emoji = "🙂" if value else "sorry"
                st.write(f"{emoji} {key}: {value}")