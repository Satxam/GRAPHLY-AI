import streamlit as st
import os
import re
import pandas as pd
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Graphly-AI", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Graphly-AI")
st.sidebar.markdown("Intelligence System")

mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Mode", "Preloaded Mode"]
)

# ---------------- HEADER ----------------
st.title("Graphly-AI: Intelligence System")
st.caption("AI-powered analysis of development reports")

# ---------------- HELPER FUNCTION ----------------
def extract_numbers(text):
    lines = text.split("\n")
    data = []

    for line in lines:
        numbers = re.findall(r'\d+\.\d+|\d+', line)
        if numbers:
            data.append((line, [float(n) for n in numbers]))

    return data

# ---------------- LLM FUNCTION ----------------
def generate_answer(context, query):
    prompt = f"""
        You are an AI assistant specialized in development reports.

        Tasks:
        - Answer clearly
        - If summarizing → give structured summary
        - Highlight key insights
        - Extract key numerical metrics from the report in JSON format like:
        {{ "metric_name": value }}

        Context:
        {context}

        Question:
        {query}
        """
    client = Groq(api_key="Your Groq API Key")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# =========================
# 🟢 UPLOAD MODE
# =========================
if mode == "Upload Mode":

    uploaded_file = st.file_uploader("Upload Development Report (PDF)", type="pdf")

    if uploaded_file is not None:

        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        st.success("PDF uploaded successfully")

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Embeddings + DB (ONLY ONCE)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)

        # Input Section
        col1, col2 = st.columns([2, 1])

        with col1:
            query = st.text_input("Ask a question")

            if st.button("Summarize Report"):
                query = "Summarize the report with key insights"

            if query:
                with st.spinner("Processing..."):

                    docs = db.similarity_search(query, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    answer = generate_answer(context, query)

                st.metric("Chunks Retrieved", len(docs))

                st.subheader("Answer")
                st.write(answer)

                # Charts
                combined_text = " ".join([doc.page_content for doc in docs])
                data = extract_numbers(combined_text)

                if data:
                    labels = [d[0][:30] for d in data]
                    values = [sum(d[1]) for d in data]

                    df = pd.DataFrame({
                        "Category": labels,
                        "Value": values
                    })

                    with col2:
                        st.subheader("Insights")
                        st.bar_chart(df.set_index("Category"))
                        st.line_chart(df.set_index("Category"))
                else:
                    st.info("No numerical data found")

                # Context
                with st.expander("View Retrieved Context"):
                    for i, doc in enumerate(docs):
                        st.write(f"Chunk {i+1}")
                        st.write(doc.page_content)
                        st.markdown("---")

        # Cleanup
        os.remove("temp.pdf")

# =========================
# 🟢 PRELOADED MODE
# =========================
elif mode == "Preloaded Mode":

    st.info("Using preloaded knowledge base")

    # ✅ ADD THIS CHECK HERE
    if not os.path.exists("vector_store"):
        st.warning("No preloaded data found. Please upload a PDF.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input("Ask a question")

        if st.button("Summarize Report"):
            query = "Summarize the report with key insights"

        if query:
            with st.spinner("Processing..."):

                docs = db.similarity_search(query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])

                answer = generate_answer(context, query)

            st.metric("Chunks Retrieved", len(docs))

            st.subheader("Answer")
            st.write(answer)

            # Charts
            combined_text = " ".join([doc.page_content for doc in docs])
            data = extract_numbers(combined_text)

            if data:
                labels = [d[0][:30] for d in data]
                values = [sum(d[1]) for d in data]

                df = pd.DataFrame({
                    "Category": labels,
                    "Value": values
                })

                with col2:
                    st.subheader("Insights")
                    st.bar_chart(df.set_index("Category"))
                    st.line_chart(df.set_index("Category"))

            else:
                st.info("No numerical data found")

            # Context
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(docs):
                    st.write(f"Chunk {i+1}")
                    st.write(doc.page_content)
                    st.markdown("---")