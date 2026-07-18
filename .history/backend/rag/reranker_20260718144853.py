import math
from collections import Counter

def compute_tfidf(chunks: list[str], query: str) -> list[float]:
    query_terms = query.lower().split()
    N = len(chunks)
    scores = []

    for chunk in chunks:
        words = chunk.lower().split()
        word_count = Counter(words)
        tf_idf_score = 0

        for term in query_terms:
            tf = word_count.get(term, 0) / (len(words) + 1)
            doc_freq = sum(1 for c in chunks if term in c.lower())
            idf = math.log((N + 1) / (doc_freq + 1)) + 1
            tf_idf_score += tf * idf

        scores.append(tf_idf_score)

    return scores

def rerank_chunks(chunks: list[dict], query: str, top_k=3) -> list[dict]:  # reduced from 6 to 3
    texts = [c["text"] for c in chunks]
    scores = compute_tfidf(texts, query)

    for i, chunk in enumerate(chunks):
        chunk["score"] = round(scores[i], 4)

    ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]