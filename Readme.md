# Multimodal RAG Assistant

A Multimodal Retrieval-Augmented Generation (RAG) system that can answer questions from PDF documents containing text, tables, charts, images, and diagrams.

## Features

- Extracts text from PDF documents
- Extracts tables and preserves their structure
- Extracts images, charts, and diagrams
- Generates textual summaries of visual content using a Vision LLM
- Creates embeddings using HuggingFace
- Stores embeddings in Pinecone
- Performs semantic similarity search
- Uses retrieved context to generate grounded answers
- Routes visual questions to a Vision LLM
- Provides a Streamlit chat interface

## Architecture

PDF
↓
PyMuPDF
↓
Text / Tables / Visuals
↓
Visual Summaries
↓
HuggingFace Embeddings
↓
Pinecone
↓
User Query
↓
Semantic Retrieval
↓
Text LLM / Vision LLM
↓
Final Answer

## Tech Stack

- Python
- LangChain
- PyMuPDF
- Pandas
- HuggingFace Embeddings
- Pinecone
- Groq
- Vision LLM
- Streamlit

## Project Structure

```text
multimodal_RAG/
│
├── app.py
├── ingest.py
├── rag.py
├── requirements.txt
├── .gitignore
└── README.md