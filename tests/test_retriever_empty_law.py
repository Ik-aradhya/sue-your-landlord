import sys
from types import SimpleNamespace

sys.path.append(".")
sys.path.append("backend")

from backend.rag import retriever


class EmptyLawIndex:
    def query(self, **kwargs):
        return {"matches": []}


class EmptyLeaseIndex:
    def query(self, **kwargs):
        return SimpleNamespace(matches=[])


def test_empty_law_matches_are_treated_as_index_unavailable(monkeypatch):
    monkeypatch.setattr(retriever, "get_law_index", lambda: EmptyLawIndex())

    import core.database as database

    monkeypatch.setattr(database, "get_pinecone_index", lambda: EmptyLeaseIndex())

    result = retriever.retrieve_chunks(
        query_vector=[0.0] * 384,
        state="maharashtra",
        session_id="test-session",
    )

    assert result["error_type"].startswith("LAW_INDEX_EMPTY_OR_UNAVAILABLE")
    assert result["law_chunks"] == []
