# MindQuery — AI-Powered Mental Health Research Assistant

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF research documents and ask natural-language questions, receiving answers grounded in the document with page-level source citations.

**Live demo:** https://huggingface.co/spaces/Dipali16/MindQuery

---

## Overview

Large language models are powerful but have three core limitations: a fixed knowledge cutoff, a limited context window, and a tendency to hallucinate when they don't know an answer. MindQuery addresses all three using RAG — instead of asking an LLM to "remember" a document, it retrieves only the most relevant sections at query time and grounds the response in that retrieved context.

MindQuery was built to fill a specific portfolio gap: moving from notebook-style ML projects to a deployed, end-to-end LLM application involving document processing, vector search, and retrieval-augmented generation.

---

## How It Works

1. **Document ingestion** — A PDF is loaded with `PyPDFLoader` and split into overlapping chunks (500 characters, 50 character overlap) using `RecursiveCharacterTextSplitter`, preventing context loss at chunk boundaries.
2. **Embedding** — Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` HuggingFace sentence-transformer model.
3. **Vector storage** — All chunk embeddings are stored in a FAISS index for fast similarity search.
4. **Retrieval** — At query time, the user's question is embedded with the same model, and FAISS retrieves the top-3 most semantically similar chunks.
5. **Generation** — The retrieved chunks are combined with the question into a prompt and sent to Groq's LLaMA 3.3 70B model, which generates an answer grounded in the retrieved context.
6. **Source attribution** — The UI displays which page(s) the answer was drawn from, so users can verify the response against the original document.

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| LLM | Groq API — LLaMA 3.3 70B |
| Embeddings | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | FAISS |
| PDF parsing | `pypdf` |
| UI | Streamlit |
| Deployment | Hugging Face Spaces (Docker SDK) |

---

## Project Structure

```
MindQuery/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Core RAG logic (chunking, embeddings, retrieval, generation)
├── requirements.txt
└── .gitignore
```

---

## Running Locally

```bash
git clone https://github.com/Dipali-cpu/MindQuery-RAG-Mental-Health-Asistant.git
cd MindQuery-RAG-Mental-Health-Asistant

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run app.py
```

---

## Key Learnings & Debugging Notes

This section documents real issues encountered during development and deployment — useful both as a personal reference and to demonstrate the debugging process.

**Query-context mismatch in RAG.** When a question's phrasing didn't match the document's phrasing (e.g., asking "according to WHO" when the retrieved chunk didn't contain the word "WHO"), the LLM would hedge and respond "I could not find this in the document" — even when the relevant information was present in the retrieved context. Rephrasing the question to match the document's own language resolved this. This is a well-documented RAG challenge, often addressed in production systems through query rewriting or expansion.

**LangChain version compatibility.** Recent LangChain versions split core functionality across `langchain`, `langchain-core`, and `langchain-community`. Imports such as `RetrievalQA` and `HumanMessage` moved between these packages across releases, requiring the RAG chain to be built manually (retrieve → build context → construct prompt → call LLM) rather than relying on a single pre-built chain. This also provided more control over prompt engineering.

**403 error on file upload (Hugging Face Docker deployment).** File uploads failed with a persistent `AxiosError 403` on the deployed Space, despite working correctly locally. The root cause: a `.streamlit/config.toml` file with CORS/XSRF settings was added to the repository root, but the Dockerfile only copies `requirements.txt` and the `src/` directory into the image — so the config file was never applied. The fix was to set the relevant settings as environment variables directly in the Dockerfile:
```dockerfile
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false
```

---

## Possible Future Improvements

- Support for multiple documents in a single session (multi-document corpus)
- Persistent storage of vector indices to avoid re-processing on every query
- Query rewriting to reduce phrasing-mismatch failures
- Citation highlighting within the source text

---

## Author

Dipali Chothmal Pawar
[GitHub](https://github.com/Dipali-cpu) · [LinkedIn](https://linkedin.com/in/dipali-chothmal-5731b032b) · [Hugging Face](https://huggingface.co/Dipali16)
