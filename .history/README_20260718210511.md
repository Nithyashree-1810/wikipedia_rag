# 📖 Wikipedia RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers questions using only Wikipedia as its knowledge source — no hallucinations, full source transparency.

Built with **FastAPI**, **Groq (Llama 3.1 8B)**, and a custom **TF-IDF reranking pipeline**.

---

## 🎯 What It Does

Instead of relying on an LLM's training data (which can hallucinate), this system:

1. Takes a user's question
2. Searches Wikipedia in real-time for the most relevant articles
3. Splits articles into smart overlapping chunks
4. Reranks chunks using TF-IDF cosine similarity
5. Passes only the top chunks as context to Llama 3.1
6. Returns a grounded answer with clickable Wikipedia source links

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Wikipedia Search API ──► Fetch Top 3 Articles
    │
    ▼
Paragraph-Aware Chunker (400 chars, 50 overlap)
    │
    ▼
TF-IDF Cosine Reranker ──► Top 3 Chunks Selected
    │
    ▼
Groq (Llama 3.1 8B) ──► Grounded Answer
    │
    ▼
Response + Wikipedia Source Links
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11 |
| LLM | Llama 3.1 8B via Groq (free tier) |
| Retrieval | Wikipedia REST API |
| Ranking | Custom TF-IDF + Cosine Similarity |
| Caching | In-memory TTL cache |
| Frontend | Vanilla JS + CSS |
| Testing | Pytest, pytest-asyncio, httpx |
| Deployment | Docker, Docker Compose |

---

## ✨ Features

- 🔍 **Real-time Wikipedia retrieval** — no stale training data
- 📄 **Paragraph-aware chunking** with overlap for better context
- 📊 **TF-IDF cosine reranking** — most relevant passages rise to the top
- 💬 **Multi-turn conversation history** — remembers last 2 turns
- ⚡ **In-memory TTL caching** — LLM called only once per unique query
- 🔁 **Fallback search** — retries with simplified query if first search fails
- 🧪 **13 pytest tests** with full mock coverage
- 🐳 **Dockerized** and ready to deploy
- 🔗 **Clickable source links** — every answer cites Wikipedia

---

## 📁 Project Structure

```
wikipedia-rag-app/
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, endpoints
│   ├── config.py                # Settings via pydantic-settings
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py         # Wikipedia search + article fetch
│   │   ├── chunker.py           # Paragraph-aware text chunking
│   │   ├── embedder.py          # TF-IDF vectorizer + cosine similarity
│   │   └── reranker.py          # Chunk reranking pipeline
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py               # Groq LLM service
│   │   └── cache.py             # In-memory TTL cache
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic request/response models
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Structured logging
├── frontend/
│   ├── index.html               # Chat UI
│   ├── app.js                   # Frontend logic
│   └── style.css                # Styling
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest path config
│   ├── test_rag.py              # Unit tests for RAG pipeline
│   └── test_api.py              # Integration tests for API endpoints
├── .env                         # API keys (not committed)
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/wikipedia-rag-app.git
cd wikipedia-rag-app
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your **free** Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
uvicorn backend.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🐳 Run with Docker

```bash
docker-compose up --build
```

App will be available at [http://localhost:8000](http://localhost:8000)

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only API tests
pytest tests/test_api.py -v

# Run only RAG unit tests
pytest tests/test_rag.py -v
```

### Test Coverage

| Test | What It Validates |
|---|---|
| `test_health_endpoint` | Server is live and responding |
| `test_root_serves_html` | Frontend is served correctly |
| `test_ask_success` | Full RAG pipeline returns correct answer |
| `test_ask_returns_session_id` | Session tracking works |
| `test_ask_with_history` | Multi-turn conversation history works |
| `test_ask_caching` | Cache hit on 2nd request, LLM called once |
| `test_ask_no_context_returns_404` | Empty Wikipedia result handled gracefully |
| `test_ask_missing_question_field` | Bad input rejected with 422 |
| `test_ask_retrieval_failure` | Wikipedia failure returns 502 |

---

## 🔌 API Reference

### `GET /health`
Returns server status and cache stats.

```json
{
  "status": "ok",
  "cache": { "cached_entries": 3 }
}
```

### `POST /ask`
Submit a question and get a Wikipedia-grounded answer.

**Request:**
```json
{
  "question": "What is quantum entanglement?",
  "session_id": "optional-session-id",
  "history": [
    { "role": "user", "content": "Previous question" },
    { "role": "assistant", "content": "Previous answer" }
  ]
}
```

**Response:**
```json
{
  "answer": "Quantum entanglement is...",
  "sources": [
    {
      "title": "Quantum entanglement",
      "url": "https://en.wikipedia.org/wiki/Quantum_entanglement"
    }
  ],
  "chunks_used": 3,
  "cached": false,
  "session_id": "abc-123"
}
```

---

## ⚠️ Limitations

- **Live events** — Wikipedia may not reflect very recent events (sports results, elections, breaking news). Always verify from official sources for time-sensitive queries.
- **Groq free tier** — Limited to 6000 tokens per minute. Context is truncated to stay within limits.
- **Wikipedia only** — Answers are strictly limited to Wikipedia content. Queries about niche or very recent topics may return no results.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👩‍💻 Author

**Nithya**
AI & Data Science Graduate 2026
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)