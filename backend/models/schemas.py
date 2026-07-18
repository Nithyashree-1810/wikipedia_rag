# models/schemas.py
from pydantic import BaseModel
from typing import Optional

class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    history: Optional[list[Message]] = []

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int
    cached: bool
    session_id: str