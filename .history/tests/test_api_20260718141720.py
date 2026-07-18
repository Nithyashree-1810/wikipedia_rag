import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app

# ─── Mock Data ───────────────────────────────────────────────────────────────

MOCK_CHUNKS = [
    {"text": "Python is a high-level programming language created by Guido van Rossum.", "title": "Python"},
    {"text": "Python supports multiple programming paradigms including procedural and object-oriented.", "title": "Python"},
    {"text": "Python was first released in 1991 and has grown in popularity.", "title": "Python"},
]

MOCK_SOURCES = [
    {"title": "Python", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}
]

MOCK_ANSWER = "Python is a high-level programming language created by Guido van Rossum in 1991."

# ─── Fixtures ─────────────────────────────────────────────────────────────────

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

# ─── Health Check ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cache" in data

# ─── Root / UI ────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_root_serves_html(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

# ─── /ask endpoint ────────────────────────────────────────────────────────────

@pytest.mark.anyio
@patch("backend.main.retrieve_context", new_callable=AsyncMock)
@patch("backend.main.generate_answer")
async def test_ask_success(mock_generate, mock_retrieve, client):
    mock_retrieve.return_value = (MOCK_CHUNKS, MOCK_SOURCES)
    mock_generate.return_value = MOCK_ANSWER

    response = await client.post("/ask", json={"question": "What is Python?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "chunks_used" in data
    assert "session_id" in data
    assert isinstance(data["cached"], bool)
    assert data["answer"] == MOCK_ANSWER
    assert data["chunks_used"] > 0


@pytest.mark.anyio
@patch("backend.main.retrieve_context", new_callable=AsyncMock)
async def test_ask_no_context_returns_404(mock_retrieve, client):
    mock_retrieve.return_value = ([], [])

    response = await client.post("/ask", json={"question": "xyznonexistenttopic123"})
    assert response.status_code == 404
    assert "No Wikipedia content found" in response.json()["detail"]


@pytest.mark.anyio
async def test_ask_empty_question(client):
    response = await client.post("/ask", json={"question": ""})
    assert response.status_code in [400, 422, 404]


@pytest.mark.anyio
async def test_ask_missing_question_field(client):
    response = await client.post("/ask", json={})
    assert response.status_code == 422


@pytest.mark.anyio
@patch("backend.main.retrieve_context", new_callable=AsyncMock)
@patch("backend.main.generate_answer")
async def test_ask_returns_session_id(mock_generate, mock_retrieve, client):
    mock_retrieve.return_value = (MOCK_CHUNKS, MOCK_SOURCES)
    mock_generate.return_value = MOCK_ANSWER

    response = await client.post("/ask", json={
        "question": "What is Python?",
        "session_id": "test-session-123"
    })
    assert response.status_code == 200
    assert response.json()["session_id"] == "test-session-123"


@pytest.mark.anyio
@patch("backend.main.retrieve_context", new_callable=AsyncMock)
@patch("backend.main.generate_answer")
async def test_ask_with_history(mock_generate, mock_retrieve, client):
    mock_retrieve.return_value = (MOCK_CHUNKS, MOCK_SOURCES)
    mock_generate.return_value = MOCK_ANSWER

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
@patch("backend.main.retrieve_context", new_callable=AsyncMock)
@patch("backend.main.generate_answer")
async def test_ask_caching(mock_generate, mock_retrieve, client):
    mock_retrieve.return_value = (MOCK_CHUNKS, MOCK_SOURCES)
    mock_generate.return_value = MOCK_ANSWER

    payload = {"question": "What is Python caching test?"}

    r1 = await client.post("/ask", json=payload)
    r2 = await client.post("/ask", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["cached"] == False
    assert r2.json()["cached"] == True
    # LLM should only be called once (second hit is from cache)
    assert mock_generate.call_count == 1


@pytest.mark.anyio
@patch("backend.main.retrieve_context", new_callable=AsyncMock)
async def test_ask_retrieval_failure(mock_retrieve, client):
    mock_retrieve.side_effect = Exception("Wikipedia API down")

    response = await client.post("/ask", json={"question": "What is Python?"})
    assert response.status_code == 502
    assert "retrieval failed" in response.json()["detail"].lower()