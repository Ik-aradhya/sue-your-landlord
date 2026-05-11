from typing import Optional

from backend.core.database import get_law_collection, get_lease_collection, get_embedding
from backend.core.config import settings
from backend.models.schemas import Chunk, RetrievedContext, Confidence, DocType


def embed_query(question: str) -> dict:
    """
    Converts user question into a query vector.

    Input:  question string
    Output: { query_vector: list[float], error_type: str | None }

    Guarantee: uses same embedding model as ingestion.
                if error_type is None → vector is valid for retrieval.
    """

    if not question or len(question.strip()) == 0:
        return {
            "query_vector": None,
            "error_type": "EMPTY_QUERY"
        }

    if len(question.strip()) > 500:
        return {
            "query_vector": None,
            "error_type": "QUERY_TOO_LONG"
        }

    try:
        vector = get_embedding(question.strip())
        return {
            "query_vector": vector,
            "error_type": None
        }

    except Exception as e:
        return {
            "query_vector": None,
            "error_type": f"EMBEDDING_FAILED: {str(e)}"
        }


def retrieve_chunks(
    query_vector: list[float],
    state: str,
    session_id: str
) -> dict:
    """
    Queries both ChromaDB collections in parallel.

    Input:  query_vector, state (for law filter), session_id (for lease filter)
    Output: { law_chunks, lease_chunks, law_score, lease_score, error_type }

    Guarantee: law_chunks contain only state-relevant law.
                lease_chunks contain only this user's lease.
                never mixes collections.
    """

    law_chunks = []
    lease_chunks = []
    law_score = 1.0       # worst possible distance — improved if results found
    lease_score = 1.0

    # ── Step 1: Query law collection ──────────────────────────────
    try:
        law_collection = get_law_collection()

        # FIX — clamp n_results to actual collection size to avoid ChromaDB crash
        law_count = law_collection.count()
        if law_count == 0:
            return {
                "law_chunks": [],
                "lease_chunks": [],
                "law_score": 1.0,
                "lease_score": 1.0,
                "error_type": "LAW_COLLECTION_EMPTY"
            }

        n_results_law = max(1, min(settings.TOP_K_LAW, law_count))

        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": n_results_law,
            "include": ["documents", "metadatas", "distances"]
        }
        if state:
            query_kwargs["where"] = {"state": state}

        law_results = law_collection.query(**query_kwargs)

        # ChromaDB returns nested lists — [0] unwraps the first query
        for i, doc in enumerate(law_results["documents"][0]):
            metadata = law_results["metadatas"][0][i] or {}
            distance = law_results["distances"][0][i]

            chunk = Chunk(
                chunk_id=f"law_{i}",
                text=doc,
                source=DocType.LAW,
                section=metadata.get("section", "Unknown Section"),
                state=metadata.get("state", state),
                session_id=None,
                start_index=0,
                end_index=len(doc)
            )
            law_chunks.append(chunk)

        # Best score = lowest distance (most similar)
        if law_results["distances"][0]:
            law_score = min(law_results["distances"][0])

    except Exception as e:
        # Law retrieval failed — return error, do not continue
        return {
            "law_chunks": [],
            "lease_chunks": [],
            "law_score": 1.0,
            "lease_score": 1.0,
            "error_type": f"LAW_RETRIEVAL_FAILED: {str(e)}"
        }

    # ── Step 2: Query lease collection ───────────────────────────
    try:
        lease_collection = get_lease_collection()
        
        # --- DEBUG CHECK ---
        all_lease = lease_collection.get(include=["metadatas"])
        print(f"\n[DEBUG] INCOMING session_id from /chat: {session_id}")
        print(f"[DEBUG] All session_ids in lease collection (showing up to 10):")
        unique_sessions = set()
        if all_lease and "metadatas" in all_lease and all_lease["metadatas"]:
            for m in all_lease["metadatas"]:
                if m:
                    unique_sessions.add(m.get("session_id"))
        for s in list(unique_sessions)[:10]:
            print(f"  → {s}")
        print("--------------------------------------------------\n")

        if session_id:
            # FIX — count only chunks belonging to this session before querying
            # ChromaDB crashes if n_results > number of matching documents
            session_count = lease_collection.count()

            if session_count > 0:
                # Get actual count for this session_id
                session_matches = lease_collection.get(
                    where={"session_id": session_id},
                    include=["documents"]
                )
                actual_count = len(session_matches["documents"])

                if actual_count > 0:
                    # Chroma rejects n_results=0; misconfigured TOP_K_LEASE must not empty lease silently
                    n_results_lease = max(1, min(settings.TOP_K_LEASE, actual_count))

                    lease_results = lease_collection.query(
                        query_embeddings=[query_vector],
                        where={"session_id": session_id},
                        n_results=n_results_lease,
                        include=["documents", "metadatas", "distances"]
                    )

                    for i, doc in enumerate(lease_results["documents"][0]):
                        metadata = lease_results["metadatas"][0][i] or {}
                        distance = lease_results["distances"][0][i]

                        chunk = Chunk(
                            chunk_id=f"lease_{i}",
                            text=doc,
                            source=DocType.LEASE,
                            section=metadata.get("section", "Lease Clause"),
                            state=None,
                            session_id=session_id,
                            start_index=0,
                            end_index=len(doc)
                        )
                        lease_chunks.append(chunk)

                    if lease_results["distances"][0]:
                        lease_score = min(lease_results["distances"][0])

    except Exception as e:
        # FIX — log the real error so it is never silently swallowed
        print(f"[LEASE RETRIEVAL ERROR] session_id={session_id} | error={str(e)}")
        lease_chunks = []
        lease_score = 1.0

    return {
        "law_chunks": law_chunks,
        "lease_chunks": lease_chunks,
        "law_score": law_score,
        "lease_score": lease_score,
        "error_type": None
    }


def compute_confidence(
    law_score: float,
    lease_score: float,
    law_chunks: list,
    lease_chunks: list
) -> Confidence:
    """
    Scores retrieval quality and decides if system should answer or fallback.

    Input:  distance scores + chunk lists
    Output: Confidence enum (HIGH | MEDIUM | LOW)

    Remember: ChromaDB returns DISTANCE not similarity.
    Lower distance = more similar = better match.

    Thresholds (from config):
        STRONG match: distance <= 0.25
        WEAK match:   distance <= 0.50
        NO match:     distance >  0.50  → fallback
    """

    law_present = len(law_chunks) > 0
    lease_present = len(lease_chunks) > 0

    # No law chunks at all — cannot answer legal questions
    if not law_present:
        return Confidence.LOW

    strong = settings.CONFIDENCE_STRONG   # 0.25
    weak   = settings.CONFIDENCE_WEAK     # 0.50

    law_strong   = law_score <= strong
    law_weak     = law_score <= weak
    lease_strong = lease_score <= strong

    # HIGH: strong law match + lease also found and relevant
    if law_strong and lease_present and lease_strong:
        return Confidence.HIGH

    # HIGH: strong law match alone is enough
    if law_strong:
        return Confidence.HIGH

    # MEDIUM: law found but not a strong match
    if law_weak:
        return Confidence.MEDIUM

    # LOW: law retrieval is too weak to trust
    return Confidence.LOW


def run_retrieval(
    question: str,
    state: str,
    session_id: str,
    retrieval_query: Optional[str] = None,
) -> dict:
    """
    Master function: runs the full retrieval pipeline.
    embed → retrieve → score confidence

    This is the only function chain.py needs to call.

    If retrieval_query is set, it is used for embedding only (e.g. history-augmented
    query). The user's literal question is still passed separately for prompting.

    Output: {
        context: RetrievedContext | None,
        should_fallback: bool,
        error_type: str | None
    }
    """

    # Stage 1 — Embed for vector search (optional context-augmented string)
    embed_text = (retrieval_query if retrieval_query is not None else question).strip()
    if len(embed_text) > 500:
        embed_text = embed_text[-500:].strip()

    embedding_result = embed_query(embed_text)
    if embedding_result["error_type"]:
        return {
            "context": None,
            "should_fallback": True,
            "error_type": embedding_result["error_type"]
        }

    # Stage 2 — Retrieve from both collections
    retrieval_result = retrieve_chunks(
        query_vector=embedding_result["query_vector"],
        state=state,
        session_id=session_id
    )
    if retrieval_result["error_type"]:
        return {
            "context": None,
            "should_fallback": True,
            "error_type": retrieval_result["error_type"]
        }

    # Stage 3 — Score confidence
    confidence = compute_confidence(
        law_score=retrieval_result["law_score"],
        lease_score=retrieval_result["lease_score"],
        law_chunks=retrieval_result["law_chunks"],
        lease_chunks=retrieval_result["lease_chunks"]
    )

    context = RetrievedContext(
        law_chunks=retrieval_result["law_chunks"],
        lease_chunks=retrieval_result["lease_chunks"],
        law_score=retrieval_result["law_score"],
        lease_score=retrieval_result["lease_score"],
        confidence=confidence
    )

    return {
        "context": context,
        "should_fallback": confidence == Confidence.LOW,
        "error_type": None
    }