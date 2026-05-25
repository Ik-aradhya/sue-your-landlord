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


def test_format_response_enriches_legal_basis_with_law_section_title():
    law_chunk = Chunk(
        chunk_id="law-8",
        text="""
        8. Court may fix standard rent and permitted increase in certain cases.
        (1) Subject to the provisions of this Act, the court may fix the
        standard rent and permitted increases.
        """,
        source=DocType.LAW,
        section="Maharashtra Rent Control Act, 1999, Section 8",
        state="maharashtra",
        start_index=0,
        end_index=180,
    )

    response = format_response(
        """
ANSWER:
Yes, you can dispute the increase in court.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 8

LEASE REFERENCE:
No clause in this lease directly addresses rent increase.

EXPLANATION:
Section 8 lets the court fix standard rent and permitted increases in disputes.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        law_chunks=[law_chunk],
    )

    assert response.legal_basis == (
        "Maharashtra Rent Control Act, 1999, Section 8 - "
        "Court may fix standard rent and permitted increase in certain cases"
    )


def test_format_response_adds_relevant_section_used_in_explanation():
    law_chunks = [
        Chunk(
            chunk_id="law-11",
            text="""
            11. Increase in rent annually and on account of improvement, etc.
            A landlord shall be entitled to make an increase of four per cent
            per annum in the rent of the premises.
            """,
            source=DocType.LAW,
            section="Maharashtra Rent Control Act, 1999, Section 11",
            state="maharashtra",
            start_index=0,
            end_index=170,
        ),
        Chunk(
            chunk_id="law-8",
            text="""
            8. Court may fix standard rent and permitted increase in certain cases.
            The court may determine the amount of permitted increase where there is a dispute.
            """,
            source=DocType.LAW,
            section="Maharashtra Rent Control Act, 1999, Section 8",
            state="maharashtra",
            start_index=0,
            end_index=160,
        ),
    ]

    response = format_response(
        """
ANSWER:
The landlord may increase rent by 4 per cent per annum.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
No clause in this lease directly addresses rent increase.

EXPLANATION:
Section 11 gives the annual increase rule. Section 8 says the court may determine the amount of permitted increase if there is a dispute.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        law_chunks=law_chunks,
        question="My landlord is asking to pay extra this month",
    )

    assert (
        "Maharashtra Rent Control Act, 1999, Section 11 - "
        "Increase in rent annually and on account of improvement, etc"
    ) in response.legal_basis
    assert (
        "Maharashtra Rent Control Act, 1999, Section 8 - "
        "Court may fix standard rent and permitted increase in certain cases"
    ) in response.legal_basis


def test_extra_charge_answer_is_tenant_protective_not_landlord_led():
    response = format_response(
        """
ANSWER:
The landlord is entitled to make an increase of 4 per cent per annum in the rent of the premises, but any dispute regarding the amount of permitted increase shall be determined by the court.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
No clause in this lease directly addresses rent increase.

EXPLANATION:
Section 11 allows an annual increase of 4 per cent per annum. If there is a dispute, the court may determine the permitted increase.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        question="My landlord is asking to pay extra this month",
    )

    assert response.answer.startswith("You do not have to pay an unsupported extra rent demand.")
    assert "limits ordinary rent increases to 4 per cent per annum" in response.answer
    assert not response.answer.startswith("The landlord is entitled")


def test_rent_increase_answer_is_limit_framed_not_landlord_led():
    response = format_response(
        """
ANSWER:
The landlord can make an increase of 4 per cent per annum in the rent of the premises.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
No clause in this lease directly addresses rent increase.

EXPLANATION:
Section 11 allows an annual increase of 4 per cent per annum.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        question="How much can rent be increased per year?",
    )

    assert response.answer == "The cited law limits ordinary rent increases to 4 per cent per annum."


def test_format_response_does_not_use_unverified_llm_section_title():
    law_chunk = Chunk(
        chunk_id="law-8-body",
        text="The court may fix standard rent and permitted increases after considering the application.",
        source=DocType.LAW,
        section="Maharashtra Rent Control Act, 1999, Section 8",
        state="maharashtra",
        start_index=0,
        end_index=100,
    )

    response = format_response(
        """
ANSWER:
Yes, you can dispute the increase in court.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 8 - Fake LLM Title

LEASE REFERENCE:
No clause in this lease directly addresses rent increase.

EXPLANATION:
Section 8 lets the court fix standard rent and permitted increases in disputes.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        law_chunks=[law_chunk],
    )

    assert response.legal_basis == "Maharashtra Rent Control Act, 1999, Section 8"


def test_irrelevant_statute_citation_is_replaced_by_lease_basis():
    law_chunk = Chunk(
        chunk_id="law-11",
        text="""
        11. Increase in rent annually and on account of improvement, etc.
        A landlord shall be entitled to make an increase of four per cent
        per annum in the rent of the premises.
        """,
        source=DocType.LAW,
        section="Maharashtra Rent Control Act, 1999, Section 11",
        state="maharashtra",
        start_index=0,
        end_index=180,
    )
    lease_chunk = Chunk(
        chunk_id="lease-security",
        text="""
        SECURITY DEPOSIT: The Tenant has paid a sum of €1,50,000/- (Rupees
        One Lakh Fifty Thousand only) as security deposit to the Landlord.
        This amount shall be refunded without interest at the end of the
        lease term, subject to deduction of any unpaid dues or damages.
        """,
        source=DocType.LEASE,
        section="Lease Document",
        start_index=0,
        end_index=260,
    )

    response = format_response(
        """
ANSWER:
Yes, your security deposit is protected by the lease refund clause.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
Security Deposit

EXPLANATION:
The legal context does not provide specific provisions for security deposit protection. The primary protection comes from the lease, which says the amount shall be refunded without interest at the end of the lease term.

CONFLICT:
NO

CONFIDENCE:
MEDIUM
""",
        Confidence.MEDIUM,
        law_chunks=[law_chunk],
        lease_chunks=[lease_chunk],
        question="Is my security deposit protected?",
    )

    assert response.legal_basis == "Lease Document - Security Deposit clause"
    assert "Section 11" not in response.legal_basis
    assert response.lease_reference == "Security Deposit"


def test_maintenance_question_uses_lease_basis_when_statute_is_generic_rent():
    law_chunk = Chunk(
        chunk_id="law-10",
        text="""
        10. Rent in excess of standard rent illegal.
        No landlord shall claim or receive any amount in excess of standard rent
        and permitted increases.
        """,
        source=DocType.LAW,
        section="Maharashtra Rent Control Act, 1999, Section 10",
        state="maharashtra",
        start_index=0,
        end_index=140,
    )
    lease_chunk = Chunk(
        chunk_id="lease-maintenance",
        text="""
        subject to deduction of any unpaid dues or damages.
        MAINTENANCE: The Tenant shall pay the monthly maintenance charges
        as levied by the Society from time to time. The Tenant shall keep
        the said premises clean.
        """,
        source=DocType.LEASE,
        section="Lease Document",
        start_index=0,
        end_index=210,
    )
    charges_chunk = Chunk(
        chunk_id="lease-other-charges",
        text="""
        ELECTRICITY & OTHER CHARGES: The Tenant shall be responsible for
        payment of electricity charges, water charges, gas charges, internet
        charges and other expenses related to the premises.
        """,
        source=DocType.LEASE,
        section="Lease Document",
        start_index=211,
        end_index=390,
    )

    response = format_response(
        """
ANSWER:
Yes, the landlord can charge maintenance separately from rent because the lease requires the tenant to pay monthly maintenance charges.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 10

LEASE REFERENCE:
Maintenance

EXPLANATION:
The lease explicitly states that the tenant must pay monthly maintenance charges. Section 10 does not prohibit this practice.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        law_chunks=[law_chunk],
        lease_chunks=[charges_chunk, lease_chunk],
        question="Can my landlord charge maintenance separately from rent?",
    )

    assert response.legal_basis == "Lease Document - Maintenance clause"
    assert "Section 10" not in response.legal_basis
    assert "Section 10" not in response.explanation
    assert "Rent Control Act" not in response.explanation
    assert response.lease_reference == "Maintenance"


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

    assert response.lease_reference == "Clause 3 — Security Deposit"


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

    assert response.lease_reference == "No direct clause found for this issue"


def test_extra_payment_question_uses_rent_increase_topic_for_lease_silence():
    lease_chunk = Chunk(
        chunk_id="lease-rent",
        text="""
        RENT: The Tenant shall pay to the Landlord a monthly rent of Rs. 38,000/-
        in advance on or before the 5th day of every month.
        """,
        source=DocType.LEASE,
        section="Lease Document",
        start_index=0,
        end_index=150,
    )

    response = format_response(
        """
ANSWER:
No, the lease does not by itself require an extra payment this month.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
No clause in this lease directly addresses extra rent this month.

EXPLANATION:
The lease states only the monthly rent amount and payment date.

CONFLICT:
NO

CONFIDENCE:
MEDIUM
""",
        Confidence.MEDIUM,
        lease_chunks=[lease_chunk],
        question="My landlord is asking to pay extra this month",
    )

    assert response.lease_reference == "No direct clause found for this issue"


def test_additional_advance_rent_question_has_clean_lease_silence_topic():
    lease_chunk = Chunk(
        chunk_id="lease-rent-advance",
        text="""
        RENT: The Tenant shall pay to the Landlord a monthly rent of Rs. 38,000/-
        in advance on or before the 5th day of every month by NEFT/Bank Transfer.
        Rent for the first month i.e. April 2026 has been paid in advance.
        """,
        source=DocType.LEASE,
        section="Lease Document",
        start_index=0,
        end_index=220,
    )

    response = format_response(
        """
ANSWER:
You do not have to pay 2 months' advance rent as the lease only specifies a monthly rent payment.
The landlord's demand for additional advance rent is not supported by the lease or the retrieved legal context.

LEGAL BASIS:
Citation: Lease Document

LEASE REFERENCE:
No clause in this lease directly addresses 2 months advance rent additionally.

EXPLANATION:
The lease specifies a monthly rent payment, and there is no provision for paying 2 months' advance rent.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        lease_chunks=[lease_chunk],
        question="Landlord is demanding 2 months advance rent additionally",
    )

    assert response.answer == (
        "You do not have to pay 2 months' advance rent as the lease only "
        "specifies a monthly rent payment. The landlord's demand for additional "
        "advance rent is not supported by the lease or the retrieved legal context."
    )
    assert response.lease_reference == "No direct clause found for this issue"


def test_unsupported_advance_rent_statute_is_removed_from_legal_basis():
    law_chunk = Chunk(
        chunk_id="law-11-advance",
        text="""
        11. Increase in rent annually and on account of improvement, etc.
        A landlord shall be entitled to make an increase of four per cent
        per annum in the rent of the premises.
        """,
        source=DocType.LAW,
        section="Maharashtra Rent Control Act, 1999, Section 11",
        state="maharashtra",
        start_index=0,
        end_index=170,
    )
    lease_chunk = Chunk(
        chunk_id="lease-rent-advance-basis",
        text="""
        RENT: The Tenant shall pay to the Landlord a monthly rent of Rs. 38,000/-
        in advance on or before the 5th day of every month by NEFT/Bank Transfer.
        Rent for the first month i.e. April 2026 has been paid in advance.
        """,
        source=DocType.LEASE,
        section="Lease Document",
        start_index=0,
        end_index=220,
    )

    response = format_response(
        """
ANSWER:
You do not have to pay 2 months' advance rent as the lease only specifies a monthly rent payment. The landlord's demand for additional advance rent is not supported by the lease or the Maharashtra Rent Control Act, 1999.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
No clause in this lease directly addresses 2 months advance rent additionally.

EXPLANATION:
The lease clearly states the monthly rent amount and the payment schedule, but it does not mention paying 2 months' advance rent. The Maharashtra Rent Control Act, 1999, Section 11, discusses permitted increases in rent, but it does not support the landlord's demand for additional advance rent.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        law_chunks=[law_chunk],
        lease_chunks=[lease_chunk],
        question="Landlord is demanding 2 months advance rent additionally",
    )

    assert response.legal_basis == "Lease Document - Rent clause"
    assert "Section 11" not in response.legal_basis
    assert "Maharashtra Rent Control Act" not in response.answer
    assert "Maharashtra Rent Control Act" not in response.explanation
    assert "retrieved legal context" in response.answer


def test_silent_lease_reference_hides_generic_source_label():
    lease_chunk = Chunk(
        chunk_id="lease-generic",
        text="""
        RENT: Monthly Rent of %38,000 (Rupees Thirty-Eight Thousand only)
        is payable by the 5% day of each month.
        """,
        source=DocType.LEASE,
        section="Lease Document",
        start_index=0,
        end_index=130,
    )

    response = format_response(
        """
ANSWER:
Yes, the landlord can increase rent only if the cited law allows it.

LEGAL BASIS:
Citation: Maharashtra Rent Control Act, 1999, Section 11

LEASE REFERENCE:
No clause in this lease directly addresses rent increase. Closest relevant clause — Lease Document: 'RENT: Monthly Rent of %38,000 is payable by the 5% day of each month.'

EXPLANATION:
The lease document does not have a specific clause addressing rent increase, but it mentions the initial monthly rent.

CONFLICT:
NO

CONFIDENCE:
HIGH
""",
        Confidence.MEDIUM,
        lease_chunks=[lease_chunk],
        question="How much can rent be increased per year?",
    )

    assert response.lease_reference == "No direct clause found for this issue"
    assert "Lease Document" not in response.lease_reference
    assert "RENT:" not in response.lease_reference
