import os
from groq import Groq
from dotenv import load_dotenv
from backend.utils.logger import logger

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a Wikipedia-grounded research assistant.
Rules:
- Answer ONLY from the provided Wikipedia context
- Be concise and accurate — 3 to 5 sentences max
- Cite sources naturally in your answer
- If the context lacks enough info, say so clearly
- For recent or ongoing events (sports, elections, news), 
  always add a disclaimer that Wikipedia may not be up to date
  and the user should verify from official sources"""
MAX_CONTEXT_CHARS = 3000   # ~750 tokens — safe for free tier

def truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n\n[Context truncated to fit token limit]"

def generate_answer(
    question: str,
    context: str,
    history: list[dict] = []
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # only last 2 turns to save tokens
    for msg in history[-2:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Wikipedia Context:\n\n{truncate_context(context)}\n\n---\n\nQuestion: {question}"
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=512,        # reduced from 1000
            temperature=0.3,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise RuntimeError(f"LLM failed: {str(e)}")