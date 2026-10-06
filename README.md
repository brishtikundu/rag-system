# RAG System

A Streamlit app that lets you chat with your PDF documents using Retrieval-Augmented Generation (LangChain + LangGraph agent, Google embeddings, Groq LLM).

## Architecture

![RAG System Architecture](docs/architecture.svg)

## Demo

![App Screenshot 1](docs/Screenshot%202026-10-06%20232803.png)

![App Screenshot 2](docs/Screenshot%202026-10-06%20233937.png)

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then add your API keys
```

## Run

Put your PDFs in `apps/docs_files/`, then:

```bash
cd apps
streamlit run rag_agent.py
```

## Stack
LangChain, LangGraph, Streamlit, PyPDF, Google Generative AI embeddings, Groq.
