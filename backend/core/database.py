from pinecone import Pinecone

from core.config import settings


# ─── Embedding ───────────────────────────────────────────────────
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def get_embedding(text: str) -> list[float]:
    result = _get_embedding_model().encode(text)
    # Pinecone requires a plain Python list — numpy arrays are not JSON-serialisable
    return result.tolist() if hasattr(result, "tolist") else list(result)


# ─── Pinecone ───────────────────────────────────────────────────
_pinecone_client: Pinecone | None = None
_law_index = None

def get_law_index():
    global _pinecone_client, _law_index
    if _law_index is None:
        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        _law_index = _pinecone_client.Index(settings.PINECONE_INDEX_NAME)
    return _law_index


def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index(settings.PINECONE_INDEX_NAME)
