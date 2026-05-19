import time

from pinecone import Pinecone

from core.config import settings


# ─── Embedding ───────────────────────────────────────────────────
_pinecone_client: Pinecone | None = None
_law_index = None
PINECONE_EMBEDDING_BATCH_SIZE = 96
PINECONE_EMBEDDING_MODEL = "llama-text-embed-v2"
PINECONE_EMBEDDING_DIMENSION = 384
DEFAULT_NAMESPACE = "__default__"


def get_pinecone_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pinecone_client


def get_embedding(text: str, input_type: str = "query") -> list[float]:
    return get_embeddings([text], input_type=input_type)[0]


def get_embeddings(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    clean_texts = [text.strip() for text in texts if text and text.strip()]
    if not clean_texts:
        return []

    embeddings: list[list[float]] = []
    for i in range(0, len(clean_texts), PINECONE_EMBEDDING_BATCH_SIZE):
        batch = clean_texts[i:i + PINECONE_EMBEDDING_BATCH_SIZE]
        embeddings.extend(_embed_with_pinecone(batch, input_type=input_type))
    return embeddings


def _embed_with_pinecone(texts: list[str], input_type: str) -> list[list[float]]:
    result = get_pinecone_client().inference.embed(
        model=PINECONE_EMBEDDING_MODEL,
        inputs=texts,
        parameters={
            "input_type": input_type,
            "truncate": "END",
            "dimension": PINECONE_EMBEDDING_DIMENSION,
        },
    )
    return _extract_embedding_values(result)


def _extract_embedding_values(result) -> list[list[float]]:
    data = getattr(result, "data", result.get("data") if isinstance(result, dict) else result)
    embeddings = []
    for item in data:
        if isinstance(item, dict):
            values = item.get("values")
        else:
            values = getattr(item, "values", None)
        if values is None:
            raise RuntimeError("Pinecone embedding response did not include vector values")
        embeddings.append([float(value) for value in values])
    return embeddings


# ─── Pinecone ───────────────────────────────────────────────────
def get_law_index():
    global _law_index
    if _law_index is None:
        _law_index = get_pinecone_client().Index(settings.PINECONE_INDEX_NAME)
    return _law_index


def get_pinecone_index():
    return get_pinecone_client().Index(settings.PINECONE_INDEX_NAME)


def lease_expires_at() -> int:
    return int(time.time() + settings.LEASE_VECTOR_TTL_MINUTES * 60)


def delete_lease_namespace(session_id: str) -> None:
    if not session_id:
        return
    get_pinecone_index().delete(delete_all=True, namespace=session_id)


def cleanup_expired_lease_namespaces() -> None:
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    namespaces = _namespace_names(stats)
    now = int(time.time())

    for namespace in namespaces:
        if namespace == DEFAULT_NAMESPACE:
            continue
        expires_at = _namespace_expires_at(index, namespace)
        if expires_at is not None and expires_at <= now:
            index.delete(delete_all=True, namespace=namespace)


def _namespace_names(stats) -> list[str]:
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
    if not namespaces:
        return []
    return list(namespaces.keys())


def _namespace_expires_at(index, namespace: str) -> int | None:
    try:
        result = index.query(
            vector=[1.0] + [0.0] * (PINECONE_EMBEDDING_DIMENSION - 1),
            namespace=namespace,
            top_k=1,
            include_metadata=True,
        )
        matches = getattr(result, "matches", None) or result.get("matches", [])
        if not matches:
            return None

        match = matches[0]
        metadata = getattr(match, "metadata", None)
        if metadata is None and isinstance(match, dict):
            metadata = match.get("metadata")
        if not metadata:
            return None

        value = metadata.get("lease_expires_at")
        return int(value) if value is not None else None
    except Exception:
        return None
