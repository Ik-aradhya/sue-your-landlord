from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from pinecone import Pinecone

from backend.core.config import settings


# ─── Embedding ───────────────────────────────────────────────────
_embedding_function = embedding_functions.DefaultEmbeddingFunction()

def get_embedding(text: str) -> list[float]:
    emb = _embedding_function([text])[0]
    return [float(x) for x in emb]


# ─── ChromaDB (lease only — temporary, session scoped) ───────────
_chroma_client = chromadb.EphemeralClient()

def get_lease_collection(session_id: str):
    return _chroma_client.get_or_create_collection(
        name=f"lease_{session_id}",
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embedding_function,
    )


# ─── Pinecone (law only — persistent, cloud) ─────────────────────
_pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
_law_index = _pinecone_client.Index(settings.PINECONE_INDEX_NAME)

def get_law_index():
    return _law_index
