import sys

sys.path.append(".")
sys.path.append("backend")

from backend.rag.chain import (
    fallback_response,
    is_retrieval_unavailable,
    retrieval_unavailable_response,
)
from backend.models.schemas import Confidence


def test_retrieval_errors_are_classified_as_unavailable():
    assert is_retrieval_unavailable("LAW_RETRIEVAL_FAILED: pinecone timeout")
    assert is_retrieval_unavailable("LAW_INDEX_EMPTY_OR_UNAVAILABLE: no law matches")
    assert is_retrieval_unavailable("EMBEDDING_FAILED: service unavailable")
    assert is_retrieval_unavailable("VECTOR_DB_ERROR: connection refused")


def test_low_confidence_fallback_remains_legal_insufficiency():
    response = fallback_response("low retrieval confidence")

    assert response.confidence == Confidence.LOW
    assert response.error_type == "FALLBACK: low retrieval confidence"
    assert "provided documents" in response.answer


def test_retrieval_unavailable_response_is_not_a_legal_answer():
    response = retrieval_unavailable_response("LAW_RETRIEVAL_FAILED: pinecone timeout")

    assert response.confidence == Confidence.LOW
    assert response.error_type.startswith("RETRIEVAL_UNAVAILABLE")
    assert "temporarily unavailable" in response.answer
    assert response.lease_reference == "Retrieval unavailable — lease and law context could not be checked"
