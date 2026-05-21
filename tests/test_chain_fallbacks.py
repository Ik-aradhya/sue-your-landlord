import sys

sys.path.append(".")
sys.path.append("backend")

from backend.rag.chain import (
    fallback_response,
    format_response,
    is_retrieval_unavailable,
    llm_unavailable_response,
    retrieval_unavailable_response,
)
from backend.models.schemas import Chunk, Confidence, DocType


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


def test_llm_unavailable_response_mentions_credits_without_legal_fallback():
    response = llm_unavailable_response("LLM_CREDITS_UNAVAILABLE")

    assert response.confidence == Confidence.LOW
    assert response.error_type == "LLM_CREDITS_UNAVAILABLE"
    assert "credits are not available" in response.answer
    assert "No legal provision was retrieved" not in response.answer


def test_format_response_rewrites_clause_number_to_lease_text():
    lease_chunk = Chunk(
        chunk_id="lease-3",
        text="""
        [Page 1]
        Clause 3 SECURITY DEPOSIT. The Tenant has paid a security deposit of
        Rs. 1,50,000/- which shall be refunded without interest at the end of
        the lease term, subject to deductions for damage.
        """,
        source=DocType.LEASE,
        section="Clause 3 - SECURITY DEPOSIT",
        start_index=0,
        end_index=180,
    )

    response = format_response(
        """
ANSWER:
Yes. The lease says the deposit is refundable at the end of the lease term.

LEGAL BASIS:
Citation: Lease Document

LEASE REFERENCE:
Clause 3

EXPLANATION:
Clause 3 covers the security deposit refund.

CONFLICT:
NOT APPLICABLE — no lease provided

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        lease_chunks=[lease_chunk],
        question="Is my security deposit protected?",
    )

    assert response.lease_reference.startswith("Clause 3 - SECURITY DEPOSIT:")
    assert "shall be refunded without interest" in response.lease_reference
    assert response.lease_reference != "Clause 3"


def test_format_response_quotes_closest_clause_when_lease_is_silent():
    lease_chunk = Chunk(
        chunk_id="lease-2",
        text="""
        [Page 1]
        Clause 2 RENT. The Tenant shall pay a monthly rent of Rs. 38,000/-
        in advance on or before the 5th day of every month.
        """,
        source=DocType.LEASE,
        section="Clause 2 - RENT",
        start_index=0,
        end_index=150,
    )

    response = format_response(
        """
ANSWER:
Yes, but only under the cited legal conditions.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
Lease Clause 2 — present but contains no provision for rent increase

EXPLANATION:
The lease states rent amount but no rent-increase mechanism.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        lease_chunks=[lease_chunk],
        question="Can my landlord increase rent?",
    )

    assert response.lease_reference.startswith(
        "No clause in this lease directly addresses rent increase."
    )
    assert "Closest relevant clause" in response.lease_reference
    assert "Clause 2 - RENT" in response.lease_reference
    assert "monthly rent of Rs. 38,000" in response.lease_reference
