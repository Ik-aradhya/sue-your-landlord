from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from backend.core.config import settings


def _resolved_chroma_path() -> str:
    """
    Chroma persistence must not depend on process cwd: uploads vs uvicorn
    started from different directories would otherwise use different stores,
    and /chat would find no lease rows for the uploaded session_id.
    """
    raw = Path(settings.CHROMA_PATH)
    if raw.is_absolute():
        return str(raw)
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / raw).resolve())


# Initialize ChromaDB client globally (uses CHROMA_PATH from settings, cwd-safe)
_chroma_client = chromadb.PersistentClient(path=_resolved_chroma_path())
_embedding_function = embedding_functions.DefaultEmbeddingFunction()

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for a given text string."""
    return _embedding_function([text])[0]

def get_law_collection():
    """Returns the law chunks collection from ChromaDB."""
    collection = _chroma_client.get_or_create_collection(
        name="law_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def get_lease_collection():
    """Returns the lease chunks collection from ChromaDB."""
    collection = _chroma_client.get_or_create_collection(
        name="lease_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    return collection
