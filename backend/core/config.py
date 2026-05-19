from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "sue-your-landlord"
    HF_TOKEN: Optional[str] = None
    LLM_MODEL: str = "llama-3.3-70b-versatile" 
    EMBEDDING_MODEL: str = "llama-text-embed-v2"
    EMBEDDING_DIMENSION: int = 384
    MAX_FILE_SIZE_MB: int = 10
    TOP_K_LAW: int = 5
    TOP_K_LEASE: int = 3
    CONFIDENCE_STRONG: float = 0.40   # distance threshold
    CONFIDENCE_WEAK: float = 0.65

    class Config:
        env_file = REPO_ROOT / ".env"

settings = Settings()
