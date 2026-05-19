import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pinecone import Pinecone

from core.config import settings


# ─── Embedding ───────────────────────────────────────────────────
_embedding_model = None
HF_EMBEDDING_BATCH_SIZE = 16
HF_EMBEDDING_TIMEOUT_SECONDS = 60


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    clean_texts = [text.strip() for text in texts]
    if not clean_texts:
        return []

    if settings.HF_TOKEN:
        return _get_hf_embeddings(clean_texts)

    results = _get_embedding_model().encode(
        clean_texts,
        batch_size=32,
        show_progress_bar=False,
    )
    if hasattr(results, "tolist"):
        return results.tolist()
    return [
        result.tolist() if hasattr(result, "tolist") else list(result)
        for result in results
    ]


def _get_hf_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), HF_EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + HF_EMBEDDING_BATCH_SIZE]
        embeddings.extend(_call_hf_feature_extraction(batch))
    return embeddings


def _call_hf_feature_extraction(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({
        "inputs": texts,
        "options": {"wait_for_model": True},
    }).encode("utf-8")

    errors: list[str] = []
    urls = [
        f"https://router.huggingface.co/hf-inference/models/{settings.EMBEDDING_MODEL}/pipeline/feature-extraction",
        f"https://api-inference.huggingface.co/models/{settings.EMBEDDING_MODEL}",
    ]
    for url in urls:
        try:
            result = _post_hf_embedding_request(url, payload)
            return _normalize_hf_embedding_response(result, expected_count=len(texts))
        except RuntimeError as e:
            errors.append(str(e))

    raise RuntimeError("Hugging Face embedding request failed: " + " | ".join(errors))


def _post_hf_embedding_request(url: str, payload: bytes):
    request = Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {settings.HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=HF_EMBEDDING_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"{url} failed: {e.reason}") from e


def _normalize_hf_embedding_response(result, expected_count: int) -> list[list[float]]:
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Hugging Face embedding request failed: {result['error']}")

    if expected_count == 1 and _is_vector(result):
        return [_to_float_vector(result)]

    if isinstance(result, list) and len(result) == expected_count:
        if all(_is_vector(item) for item in result):
            return [_to_float_vector(item) for item in result]
        if all(_is_matrix(item) for item in result):
            return [_mean_pool(item) for item in result]

    if expected_count == 1 and _is_matrix(result):
        return [_mean_pool(result)]

    raise RuntimeError("Hugging Face embedding response had an unexpected shape")


def _is_vector(value) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], (int, float))


def _is_matrix(value) -> bool:
    return isinstance(value, list) and bool(value) and _is_vector(value[0])


def _to_float_vector(vector) -> list[float]:
    return [float(value) for value in vector]


def _mean_pool(matrix) -> list[float]:
    width = len(matrix[0])
    pooled = [0.0] * width
    for row in matrix:
        for i, value in enumerate(row):
            pooled[i] += float(value)
    return [value / len(matrix) for value in pooled]


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
