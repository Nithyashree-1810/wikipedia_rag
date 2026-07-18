from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = ""
    cache_ttl_seconds: int = 3600
    max_chunks: int = 6
    chunk_size: int = 700
    chunk_overlap: int = 100
    wiki_search_limit: int = 3
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.3

    class Config:
        env_file = ".env"

settings = Settings()