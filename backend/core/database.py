from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pinecone import Pinecone

from core.config import settings


def _resolved_chroma_path() -> str:
    raw = Path(settings.CHROMA_PATH)
    if raw.is_absolute():
        return str(raw)
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / raw).resolve())


# ─── Embedding ───────────────────────────────────────────────────
_embedding_function = embedding_functions.DefaultEmbeddingFunction()

def get_embedding(text: str) -> list[float]:
    result = _embedding_function([text])[0]
    # Pinecone requires a plain Python list — numpy arrays are not JSON-serialisable
    return result.tolist() if hasattr(result, "tolist") else list(result)


# ─── ChromaDB (lease only — persistent on disk so data survives restarts) ───
_chroma_client = chromadb.PersistentClient(path=_resolved_chroma_path())

def get_lease_collection(session_id: str):
    return _chroma_client.get_or_create_collection(
        name=f"lease_{session_id}",
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embedding_function,
    )


# ─── Pinecone (law only — persistent, cloud) ─────────────────────
_pinecone_client: Pinecone | None = None
_law_index = None

def get_law_index():
    global _pinecone_client, _law_index
    if _law_index is None:
        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        _law_index = _pinecone_client.Index(settings.PINECONE_INDEX_NAME)
    return _law_index
