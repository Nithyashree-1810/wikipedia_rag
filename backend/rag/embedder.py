import math
from collections import Counter

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def build_vocab(corpus: list[str]) -> list[str]:
    vocab = set()
    for doc in corpus:
        vocab.update(tokenize(doc))
    return list(vocab)

def tf(term: str, doc: str) -> float:
    words = tokenize(doc)
    count = words.count(term)
    return count / len(words) if words else 0.0

def idf(term: str, corpus: list[str]) -> float:
    N = len(corpus)
    df = sum(1 for doc in corpus if term in tokenize(doc))
    return math.log((N + 1) / (df + 1)) + 1

def tfidf_vector(doc: str, vocab: list[str], corpus: list[str]) -> list[float]:
    return [tf(term, doc) * idf(term, corpus) for term in vocab]

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def rank_chunks_by_similarity(chunks: list[str], query: str) -> list[tuple[int, float]]:
    corpus = chunks + [query]
    vocab = build_vocab(corpus)

    query_vec = tfidf_vector(query, vocab, corpus)
    scored = []

    for i, chunk in enumerate(chunks):
        chunk_vec = tfidf_vector(chunk, vocab, corpus)
        score = cosine_similarity(query_vec, chunk_vec)
        scored.append((i, round(score, 4)))

    return sorted(scored, key=lambda x: x[1], reverse=True)