import sys

sys.path.append(".")

from backend.rag.prompts import (
    SYSTEM_PROMPT,
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
    assert "answer the notice requirement first" in guidance
    assert "Do not start with \"No\"" in guidance

    messages = build_prompt(
        question="What notice is required for a rent increase?",
        law_chunks=[],
        lease_chunks=[],
        state="maharashtra",
        history=[],
    )

    assert "RENT INCREASE CLUSTER" in messages[1]["content"]


def test_system_prompt_contains_product_answer_rules():
    assert 'Never start with "Only if"' in SYSTEM_PROMPT
    assert "No legal provision exists for this under [JURISDICTION] law." in SYSTEM_PROMPT
    assert "ANSWER WRITING QUALITY" in SYSTEM_PROMPT
    assert "not a search result summary" in SYSTEM_PROMPT
    assert "Yes/No/Partially/Likely first" in SYSTEM_PROMPT
    assert "LOW confidence must be useful" in SYSTEM_PROMPT
    assert "Do not force a Yes/No opening" in SYSTEM_PROMPT
    assert "specific statutory section" in SYSTEM_PROMPT


def test_lockout_prompt_gets_state_neutral_self_help_guidance():
    question = "My landlord changed the lock while I was at work — what can I do?"
    guidance = build_issue_guidance(question)

    assert "LOCKOUT / SELF-HELP EVICTION" in guidance
    assert "selected state" in guidance
    assert "cite that section" in guidance
    assert "Never imply a unilateral lock change" in guidance

    queries = build_retrieval_queries(question)

    assert len(queries) == 4
    assert not any("Maharashtra" in query for query in queries)
    assert any("restore possession" in query for query in queries)


def test_no_written_lease_prompt_gets_state_neutral_guidance():
    question = "I never signed a written lease — do I have any rights?"
    guidance = build_issue_guidance(question)

    assert "NO WRITTEN LEASE / ORAL TENANCY" in guidance
    assert "selected state" in guidance
    assert "deemed tenant" in guidance
    assert "based on statute" in guidance

    queries = build_retrieval_queries(question)

    assert len(queries) == 4
    assert not any("Maharashtra" in query for query in queries)
    assert any("deemed tenant" in query for query in queries)


def test_holdover_prompt_gets_state_neutral_possession_guidance():
    question = "My lease ended but landlord hasn't asked me to leave — can I keep staying?"
    guidance = build_issue_guidance(question)

    assert "LEASE ENDED / HOLDOVER POSSESSION" in guidance
    assert "selected state" in guidance
    assert "court proceedings" in guidance
    assert "Do not invent \"statutory tenant\" language" in guidance

    queries = build_retrieval_queries(question)

    assert len(queries) == 4
    assert not any("Maharashtra" in query for query in queries)
    assert any("court order" in query for query in queries)
    assert any("statutory tenant" in query for query in queries)


def test_rent_increase_uses_multi_query_retrieval():
    queries = build_retrieval_queries("How much can rent be increased per year?")

    assert len(queries) == 4
    assert any("ordinary annual rent increase" in query for query in queries)
    assert any("determine amount permitted increase" in query for query in queries)
    assert any("special additions improvements" in query for query in queries)


def test_non_rent_issue_uses_single_retrieval_query():
    queries = build_retrieval_queries("Is my security deposit refundable?")

    assert len(queries) == 1


def test_receipt_payment_prompt_gets_issue_guidance():
    guidance = build_issue_guidance("I paid rent in cash but have no receipt. Am I protected?")

    assert "RECEIPT / CASH PAYMENT CLUSTER" in guidance
    assert "landlord's statutory obligation" in guidance
    assert "do not start with \"No\"" in guidance


def test_receipt_payment_uses_multi_query_retrieval():
    queries = build_retrieval_queries("I paid rent in cash but have no receipt. Am I protected?")

    assert len(queries) == 4
    assert any("giving receipt for any amount received compulsory" in query for query in queries)
    assert any("fails to give written receipt" in query for query in queries)
    assert any("rent receipt cash payment proof" in query for query in queries)
