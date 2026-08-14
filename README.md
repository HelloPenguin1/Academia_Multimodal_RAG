# 🔬 ResearchPro — Advanced Academic RAG System

A conversational research assistant for academic PDF documents. Upload a research paper and query it with natural language — the system retrieves and synthesises relevant information using a multi-stage retrieval pipeline backed by hybrid search, cross-encoder reranking, and a conversational LLM.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       FRONTEND (Streamlit)                      │
│          streamlit_app.py  ─  REST client to FastAPI            │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (localhost:8000)
┌────────────────────────────▼────────────────────────────────────┐
│                      BACKEND (FastAPI)                          │
│                        main.py                                  │
│                                                                 │
│   POST /upload_file    POST /query    DELETE /delete            │
└──────────┬─────────────────┬───────────────────────────────────┘
           │                 │
    ┌──────▼──────┐   ┌──────▼──────────────────────────────────┐
    │  INGESTION  │   │         QUERY PIPELINE                   │
    │  PIPELINE   │   │                                          │
    │             │   │  1. Query Reformulation (Llama-3.3-70B)  │
    │ Docling PDF │   │  2. Hybrid Retrieval (BM25 + FAISS)      │
    │ Extraction  │   │  3. Cross-Encoder Reranking              │
    │     ↓       │   │  4. Answer Generation (GPT-o3-120B)      │
    │ Markdown    │   │  5. Conversational Memory                │
    │ Splitting   │   │                                          │
    │     ↓       │   └─────────────────────────────────────────┘
    │ FAISS Index │
    │ BM25 Index  │
    └─────────────┘
```

---

## 🔄 Request Lifecycle

### 1. Document Ingestion (`POST /upload_file`)

```
PDF Upload
    │
    ▼
Docling DocumentConverter
    │  Extracts text, tables, and structure-aware content
    │  Outputs a structured DoclingDocument
    ▼
export_to_markdown()
    │  Converts the DoclingDocument to Markdown text
    │  Tables are preserved as native Markdown pipe tables (| col | col |)
    │  — not flattened to prose — keeping row/cell structure retrievable
    ▼
MarkdownHeaderTextSplitter
    │  Splits on H1 / H2 / H3 headers
    │  Each chunk preserves its header metadata
    ▼
Dual Indexing (in-memory, no persistence)
    ├── FAISS Vector Store  ← HuggingFace Embeddings (bge-small-en-v1.5)
    │      k=25 nearest neighbours on similarity search
    └── BM25Retriever       ← rank-bm25, k=25
```

### 2. Query Pipeline (`POST /query`)

```
User Question  +  Chat History
        │
        ▼
┌──────────────────────────────────┐
│  Query Reformulation             │
│  Model: Llama-3.3-70B (Groq)     │
│  Expands pronouns, co-refs,      │
│  preserves entity names          │
└──────────────┬───────────────────┘
               │ Standalone query
               ▼
┌──────────────────────────────────┐
│  Hybrid Retrieval                │
│  EnsembleRetriever               │
│    BM25     40%  (keyword)       │
│    FAISS    60%  (semantic)      │
│  Returns top-25 candidates       │
└──────────────┬───────────────────┘
               │ 25 candidate chunks
               ▼
┌──────────────────────────────────┐
│  Cross-Encoder Reranking         │
│  Model: ms-marco-MiniLM-L-6-v2   │
│  (HuggingFace, runs locally)     │
│  Re-scores all 25 pairs          │
│  Returns top_n=5 final chunks    │
└──────────────┬───────────────────┘
               │ 5 high-quality context chunks
               ▼
┌──────────────────────────────────┐
│  Answer Generation               │
│  Model: openai/gpt-oss-120B      │
│         (served via Groq API)    │
│  Strict grounding prompt:        │
│  - No hallucination rules        │
│  - Mandatory inline citations    │
│  - Multi-document attribution    │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Conversational Memory           │
│  RunnableWithMessageHistory      │
│  In-memory ChatMessageHistory    │
│  Scoped by session_id (UUID)     │
└──────────────────────────────────┘
```

---

## 📁 Project Structure

```
ResearchPro_AdvancedRAG/
├── config/
│   └── config.py               # All model instantiation (LLMs, embeddings, reranker)
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app — 3 endpoints: /upload_file, /query, /delete
│   │   ├── services/
│   │   │   ├── vision_service.py     # PDF ingestion: Docling + Markdown splitting
│   │   │   ├── document_service.py   # Index creation: FAISS + BM25 + EnsembleRetriever
│   │   │   ├── reranker.py           # Cross-encoder reranking via ContextualCompressionRetriever
│   │   │   └── rag_service.py        # Full RAG chain: reformulation → retrieval → generation
│   │   └── evaluation/               # RAGAS evaluation scripts and results
│   └── utils/
│       └── session_manager.py        # In-memory chat history store (session_id → ChatMessageHistory)
│
├── frontend/
│   └── streamlit_app.py        # Streamlit UI — PDF upload + conversational chat interface
│
└── requirements.txt
```

> **Note on naming:** `vision_service.py` handles PDF text extraction via Docling — there is **no active vision/image model** in the current pipeline. The vision LLM (`llama-4-scout`) is commented out. Similarly, `MultimodalProcessor` is a plain PDF-to-Markdown processor, not a multimodal model.

---

## 🧠 Models & Components

| Role | Model / Library | Provider |
|---|---|---|
| **Answer Generation** | `openai/gpt-oss-120b` | Groq API |
| **Query Reformulation** | `llama-3.3-70b-versatile` | Groq API |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | HuggingFace (local) |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace (local) |
| **PDF Parsing** | Docling `DocumentConverter` | Local |
| **Keyword Search** | BM25 (`rank-bm25`) | Local |
| **Vector Store** | FAISS (`faiss-cpu`) | In-memory |
| **Text Splitting** | `MarkdownHeaderTextSplitter` | LangChain |
| **Orchestration** | LangChain v0.3 | — |

---

## ⚙️ Retrieval Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `k` (FAISS + BM25 candidates) | 25 each | Wide net before reranking |
| BM25 weight | 0.40 | Strong for exact model/dataset names |
| FAISS weight | 0.60 | Dominant for semantic similarity |
| `top_n` (reranker) | 5 | Final context sent to LLM |
| Reformulation LLM temp | 0.1 | Near-deterministic, preserves entities |
| Answer LLM temp | 0 | Fully deterministic, no hallucination |

---

## 💬 Conversation & Session Management

- Each Streamlit session generates a UUID (`session_id`) on page load.
- The `SessionManager` maintains a Python dict mapping `session_id → ChatMessageHistory`.
- State is **in-memory only** — sessions and the vector store are lost on server restart.
- The `DELETE /delete` endpoint clears both the FAISS/BM25 indexes and all session histories.

---

## 📊 Evaluation

RAGAS evaluation results are stored under `backend/app/evaluation/`:

```
evaluation/
├── single_doc_eval/     # Evaluation configs for single-document queries
├── single_doc_results/  # Metric outputs (Faithfulness, Recall, Relevancy)
├── multi_doc_eval/      # Evaluation configs for cross-document queries
└── mutli_doc_results/   # Multi-doc metric outputs
```
