# Graphly-AI: Intelligence System

An AI-powered platform for analyzing development reports using Retrieval-Augmented Generation (RAG), enabling users to extract insights, summaries, and visual trends from unstructured documents.

---

## Overview

Graphly-AI is designed to simplify the understanding of complex development and policy reports. It allows users to upload documents and interact with them using natural language queries, while also generating data-driven visual insights.

This project demonstrates how AI can support **knowledge extraction, evaluation, and decision-making** in development contexts.

---

## Key Features

* Upload and analyze PDF reports dynamically
* Semantic search using vector embeddings (FAISS)
* Context-aware answer generation using LLM (Groq)
* Automatic summarization of reports
* Extraction of numerical patterns for visualization
* Interactive dashboard with bar and line charts
* Dual modes:

  * Upload Mode (dynamic ingestion)
  * Preloaded Mode (fast querying)

---

## System Architecture

```
PDF → Text → Chunks → Embeddings → Vector Database
                                         ↓
User Query → Retrieval → Context → LLM → Answer + Insights
```

---

## Tech Stack

* Python
* Streamlit (UI)
* LangChain (RAG pipeline)
* FAISS (Vector database)
* HuggingFace Embeddings
* Groq API (LLM)
* Pandas (data processing)

---

## Use Case

Graphly-AI is particularly useful for:

* Development report analysis
* Policy evaluation
* Knowledge management
* Extracting insights from large documents
* Supporting data-driven decision-making

---

## Getting Started

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 3. Run the application

```
streamlit run app.py
```

---

## Deployment

This application can be deployed using **Streamlit Cloud**.

After deployment, you can access it via:

```
https://your-app.streamlit.app
```

---

## Project Highlights

* Built a full RAG pipeline from scratch
* Integrated LLM for contextual answer generation
* Designed a user-friendly dashboard for interaction
* Implemented both dynamic and precomputed retrieval systems
* Addressed real-world challenges such as model deprecation and PDF parsing

---

## Future Improvements

* Improved structured data extraction for better visual insights
* Chat-based interface with conversation memory
* Persistent storage for uploaded documents
* Enhanced evaluation metrics for extracted insights

---

## Author

Satyam

---

## Acknowledgment

This project was developed as part of exploring AI applications in development and evaluation contexts, aligning with real-world use cases such as those addressed by international organizations like IFAD.

---

## Live Demo

(Add your deployed Streamlit link here)
