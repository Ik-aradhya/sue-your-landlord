import sys

sys.path.append(".")

from backend.rag.prompts import build_retrieval_query


def test_rent_increase_query_includes_ordinary_annual_terms():
    query = build_retrieval_query("How much can rent be increased per year?")

    assert "standard rent" in query
    assert "annual increase" in query
    assert "ordinary rent increase" in query


def test_special_improvement_query_includes_special_category_terms():
    query = build_retrieval_query("Can my landlord charge for improvements?")

    assert "special additions" in query
    assert "improvements" in query
    assert "expenses" in query
