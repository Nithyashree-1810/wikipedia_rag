import pytest
from rag.chunker import chunk_by_paragraph, clean_text
from rag.reranker import rerank_chunks, compute_tfidf

def test_clean_text():
    raw = "Hello\n\n\n\nWorld  extra   spaces"
    result = clean_text(raw)
    assert "\n\n\n" not in result
    assert "  " not in result

def test_chunking():
    text = ("This is sentence one. " * 30 + "\n\n") * 5
    chunks = chunk_by_paragraph(text)
    assert len(chunks) > 0
    assert all(len(c) > 0 for c in chunks)

def test_reranker():
    chunks = [
        {"text": "Python is a programming language.", "title": "Python"},
        {"text": "The sun is a star in the solar system.", "title": "Sun"},
        {"text": "Python was created by Guido van Rossum.", "title": "Python"},
    ]
    results = rerank_chunks(chunks, "Python programming language", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]

def test_tfidf_scores():
    chunks = ["machine learning is AI", "cooking recipes and food", "deep learning neural networks"]
    scores = compute_tfidf(chunks, "machine learning")
    assert scores[0] > scores[1]