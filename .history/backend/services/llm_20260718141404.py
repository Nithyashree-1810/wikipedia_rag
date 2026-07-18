import os
from groq import Groq
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a Wikipedia-grounded research assistant.
Rules:
- Answer ONLY from the provided Wikipedia context
- Be concise, accurate, and cite sources naturally
- If the context lacks enough info, say so clearly — never guess
- Use bullet points for complex multi-part answers
- Keep answers under 200 words unless the question needs more"""

def generate_answer(
    question: str,
    context: str,
    history: list[dict] = []
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Wikipedia Context:\n\n{context}\n\n---\n\nQuestion: {question}"
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=1000,
            temperature=0.3,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise RuntimeError(f"LLM failed: {str(e)}")