import sys

sys.path.append(".")
sys.path.append("backend")

from backend.rag.chain import (
    fallback_response,
    format_response,
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
    assert "No legal provision was retrieved" in response.answer
    assert "Missing:" in response.explanation


def test_conflict_answer_with_specific_section_can_remain_high_confidence():
    response = format_response(
        """
ANSWER:
No. The repair clause is not enforceable to the extent it contradicts the statute.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 14

LEASE REFERENCE:
Clause 4

EXPLANATION:
Section 14 places the repair obligation on the landlord, so the lease cannot shift all statutory repair duties to the tenant. Dispute the repair demand in writing and cite Section 14.

CONFLICT:
YES — Clause 4 conflicts with Maharashtra Rent Control Act, 1999, Section 14.

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
    )

    assert response.conflict_flag is True
    assert response.confidence == Confidence.HIGH


def test_retrieval_unavailable_response_is_not_a_legal_answer():
    response = retrieval_unavailable_response("LAW_RETRIEVAL_FAILED: pinecone timeout")

    assert response.confidence == Confidence.LOW
    assert response.error_type.startswith("RETRIEVAL_UNAVAILABLE")
    assert "temporarily unavailable" in response.answer
    assert response.lease_reference == "Retrieval unavailable — lease and law context could not be checked"
