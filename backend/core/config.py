
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str
    LLM_MODEL: str = "llama-3.3-70b-versatile" 
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CHROMA_PATH: str = "./vectorstore"
    LAW_COLLECTION: str = "law_chunks"
    LEASE_COLLECTION: str = "lease_chunks"
    MAX_FILE_SIZE_MB: int = 10
    TOP_K_LAW: int = 5
    TOP_K_LEASE: int = 3
    CONFIDENCE_STRONG: float = 0.25   # distance threshold
    CONFIDENCE_WEAK: float = 0.50

    class Config:
        env_file = ".env"

settings = Settings()

