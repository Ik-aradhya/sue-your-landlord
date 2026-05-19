import sys

sys.path.append(".")

from backend.rag.prompts import (
    build_issue_guidance,
    build_prompt,
    build_retrieval_queries,
    build_retrieval_query,
)


def test_rent_increase_query_includes_ordinary_annual_terms():
    query = build_retrieval_query("How much can rent be increased per year?")

    assert "standard rent" in query
    assert "annual increase" in query
    assert "ordinary rent increase" in query
    assert "percentage rate cap formula" in query
    assert "court fix standard rent permitted increase" in query


def test_special_improvement_query_includes_special_category_terms():
    query = build_retrieval_query("Can my landlord charge for improvements?")

    assert "special additions" in query
    assert "improvements" in query
    assert "expenses" in query


def test_rent_increase_phrasings_share_canonical_retrieval_hint():
    amount_query = build_retrieval_query("How much can rent be increased per year?")
    notice_query = build_retrieval_query("What notice is required for a rent increase?")
    dispute_query = build_retrieval_query("Can I dispute a rent increase?")

    canonical_terms = [
        "ordinary rent increase",
        "percentage rate cap formula",
        "court fix standard rent permitted increase",
        "additions improvements repairs taxes amenities services",
    ]

    for term in canonical_terms:
        assert term in amount_query
        assert term in notice_query
        assert term in dispute_query


def test_rent_increase_prompt_gets_issue_guidance():
    guidance = build_issue_guidance("What notice is required for a rent increase?")

    assert "RENT INCREASE CLUSTER" in guidance
    assert "no specific notice period is stated" in guidance

    messages = build_prompt(
        question="What notice is required for a rent increase?",
        law_chunks=[],
        lease_chunks=[],
        state="maharashtra",
        history=[],
    )

    assert "RENT INCREASE CLUSTER" in messages[1]["content"]


def test_rent_increase_uses_multi_query_retrieval():
    queries = build_retrieval_queries("How much can rent be increased per year?")

    assert len(queries) == 4
    assert any("ordinary annual rent increase" in query for query in queries)
    assert any("determine amount permitted increase" in query for query in queries)
    assert any("special additions improvements" in query for query in queries)


def test_non_rent_issue_uses_single_retrieval_query():
    queries = build_retrieval_queries("Is my security deposit refundable?")

    assert len(queries) == 1
