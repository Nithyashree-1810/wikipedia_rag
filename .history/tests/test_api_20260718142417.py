import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock,MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from backend.main import app

MOCK_CHUNKS = [
    {"text": "Python is a high-level programming language created by Guido van Rossum.", "title": "Python"},
    {"text": "Python supports multiple programming paradigms including procedural and object-oriented.", "title": "Python"},
    {"text": "Python was first released in 1991 and has grown in popularity.", "title": "Python"},
]

MOCK_SOURCES = [
    {"title": "Python", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}
]

MOCK_ANSWER = "Python is a high-level programming language created by Guido van Rossum in 1991."

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c

# ─── Health & Root ────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cache" in data

@pytest.mark.anyio
async def test_root_serves_html(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

# ─── /ask endpoint ────────────────────────────────────────────────────────────

@pytest.mark.anyio
@patch("backend.rag.retriever.search_wikipedia", new_callable=AsyncMock)
@patch("backend.rag.retriever.fetch_article", new_callable=AsyncMock)
@patch("backend.services.llm.client")
async def test_ask_success(mock_llm, mock_fetch, mock_search, client):
    mock_search.return_value = ["Python"]
    mock_fetch.return_value = "Python is a high-level programming language. " * 20

    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_ANSWER
    mock_llm.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    response = await client.post("/ask", json={"question": "What is Python?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "chunks_used" in data
    assert "session_id" in data
    assert isinstance(data["cached"], bool)

@pytest.mark.anyio
@patch("backend.rag.retriever.search_wikipedia", new_callable=AsyncMock)
async def test_ask_no_context_returns_404(mock_search, client):
    mock_search.return_value = []
    response = await client.post("/ask", json={"question": "xyznonexistenttopic123"})
    assert response.status_code == 404
    assert "No Wikipedia content found" in response.json()["detail"]

@pytest.mark.anyio
async def test_ask_missing_question_field(client):
    response = await client.post("/ask", json={})
    assert response.status_code == 422

@pytest.mark.anyio
@patch("backend.rag.retriever.search_wikipedia", new_callable=AsyncMock)
@patch("backend.rag.retriever.fetch_article", new_callable=AsyncMock)
@patch("backend.services.llm.client")
async def test_ask_returns_session_id(mock_llm, mock_fetch, mock_search, client):
    mock_search.return_value = ["Python"]
    mock_fetch.return_value = "Python is a high-level programming language. " * 20

    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_ANSWER
    mock_llm.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    response = await client.post("/ask", json={
        "question": "What is Python?",
        "session_id": "test-session-123"
    })
    assert response.status_code == 200
    assert response.json()["session_id"] == "test-session-123"

@pytest.mark.anyio
@patch("backend.rag.retriever.search_wikipedia", new_callable=AsyncMock)
@patch("backend.rag.retriever.fetch_article", new_callable=AsyncMock)
@patch("backend.services.llm.client")
async def test_ask_with_history(mock_llm, mock_fetch, mock_search, client):
    mock_search.return_value = ["Python"]
    mock_fetch.return_value = "Python is a high-level programming language. " * 20

    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_ANSWER
    mock_llm.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    response = await client.post("/ask", json={
        "question": "Tell me more about it",
        "history": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."}
        ]
    })
    assert response.status_code == 200
    assert "answer" in response.json()

@pytest.mark.anyio
@patch("backend.rag.retriever.search_wikipedia", new_callable=AsyncMock)
@patch("backend.rag.retriever.fetch_article", new_callable=AsyncMock)
@patch("backend.services.llm.client")
async def test_ask_caching(mock_llm, mock_fetch, mock_search, client):
    mock_search.return_value = ["Python"]
    mock_fetch.return_value = "Python is a high-level programming language. " * 20

    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_ANSWER
    mock_llm.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    payload = {"question": "What is Python caching unique test query?"}
    r1 = await client.post("/ask", json=payload)
    r2 = await client.post("/ask", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["cached"] == False
    assert r2.json()["cached"] == True
    # LLM called only once, second response served from cache
    assert mock_llm.chat.completions.create.call_count == 1

@pytest.mark.anyio
@patch("backend.rag.retriever.search_wikipedia", new_callable=AsyncMock)
async def test_ask_retrieval_failure(mock_search, client):
    mock_search.side_effect = Exception("Wikipedia API down")
    response = await client.post("/ask", json={"question": "What is Python?"})
    assert response.status_code == 502
    assert "retrieval failed" in response.json()["detail"].lower()