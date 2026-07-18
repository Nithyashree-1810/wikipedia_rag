from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid, time, os
from dotenv import load_dotenv

from backend.models.schemas import QueryRequest, QueryResponse
from backend.rag.retriever import retrieve_context
from backend.rag.reranker import rerank_chunks
from backend.services.cache import cache
from backend.services.llm import generate_answer
from backend.utils.logger import logger

load_dotenv()

app = FastAPI(title="Wikipedia RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "cache": cache.stats()}

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    start = time.time()
    session_id = req.session_id or str(uuid.uuid4())

    cache_key = req.question.strip().lower()
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for: {req.question}")
        return {**cached, "cached": True, "session_id": session_id}

    try:
        raw_chunks, sources = await retrieve_context(req.question)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=502, detail="Wikipedia retrieval failed")

    if not raw_chunks:
        raise HTTPException(status_code=404, detail="No Wikipedia content found")

    top_chunks = rerank_chunks(raw_chunks, req.question, top_k=6)
    context = "\n\n---\n\n".join(
        f"[Source: {c['title']} \n{c['text']}"
        for c in top_chunks
    )

    # pass history as list of dicts
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in (req.history or [])[-4:]
    ]

    answer = generate_answer(req.question, context, history)

    elapsed = round(time.time() - start, 2)
    logger.info(f"Query answered in {elapsed}s | chunks={len(top_chunks)}")

    result = {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(top_chunks),
        "cached": False
    }
    cache.set(cache_key, result)

    return {**result, "session_id": session_id}