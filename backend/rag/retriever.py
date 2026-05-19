from typing import Optional

from core.database import get_law_index, get_embedding
from core.config import settings
from models.schemas import Chunk, RetrievedContext, Confidence, DocType


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
    Queries law and lease vectors from Pinecone.

    Input:  query_vector, state (for law filter), session_id (for lease filter)
    Output: { law_chunks, lease_chunks, law_score, lease_score, error_type }

    Guarantee: law_chunks contain only state-relevant law.
                lease_chunks contain only this user's lease.
                never mixes lease namespaces.
    """

    law_chunks = []
    lease_chunks = []
    law_score = 1.0       # worst possible distance — improved if results found
    lease_score = 1.0

    # ── Step 1: Query law collection (Pinecone) ───────────────────────
    try:
        law_index = get_law_index()

        pinecone_results = law_index.query(
            vector=query_vector,
            top_k=settings.TOP_K_LAW,
            filter={"state": state} if state else None,
            include_metadata=True
        )

        matches = pinecone_results.get("matches", [])

        if not matches:
            # Law collection returned nothing — don't bail, still try the lease
            print("[WARN] Pinecone returned no law matches for this query.")
        else:
            for i, match in enumerate(matches):
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                score = match.get("score", 0.0)
                # Pinecone returns similarity (higher = better), convert to distance
                distance = 1.0 - score

                chunk = Chunk(
                    chunk_id=f"law_{i}",
                    text=text,
                    source=DocType.LAW,
                    section=metadata.get("section", "Unknown Section"),
                    state=metadata.get("state", state),
                    session_id=None,
                    start_index=0,
                    end_index=len(text)
                )
                law_chunks.append(chunk)

            if matches:
                law_score = 1.0 - matches[0]["score"]  # best match = first result

    except Exception as e:
        return {
            "law_chunks": [],
            "lease_chunks": [],
            "law_score": 1.0,
            "lease_score": 1.0,
            "error_type": f"LAW_RETRIEVAL_FAILED: {str(e)}"
        }

    # ── Step 2: Query lease namespace ───────────────────────────
    try:
        from core.database import get_pinecone_index
        index = get_pinecone_index()
        lease_results = index.query(
            vector=query_vector,
            namespace=session_id,
            top_k=settings.TOP_K_LEASE,
            include_metadata=True
        )

        lease_chunks = []
        for match in lease_results.matches:
            text = match.metadata.get("text", "")
            chunk = Chunk(
                chunk_id=match.id,
                text=text,
                source=DocType.LEASE,
                section=match.metadata.get("section", ""),
                state=None,
                session_id=session_id,
                start_index=0,
                end_index=len(text)
            )
            lease_chunks.append(chunk)

        if lease_results.matches:
            lease_score = 1.0 - lease_results.matches[0].score

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

    Pinecone returns similarity; retrieve_chunks converts it to distance.
    Lower distance = more similar = better match.

    Thresholds (from config):
        STRONG match: distance <= 0.25
        WEAK match:   distance <= 0.50
        NO match:     distance >  0.50  → fallback
    """

    law_present   = len(law_chunks) > 0
    lease_present = len(lease_chunks) > 0

    strong = settings.CONFIDENCE_STRONG   # 0.40
    weak   = settings.CONFIDENCE_WEAK     # 0.65

    law_strong   = law_score   <= strong
    law_weak     = law_score   <= weak
    lease_strong = lease_score <= strong
    lease_weak   = lease_score <= weak

    # HIGH: strong law match + lease also found and relevant
    if law_strong and lease_present and lease_strong:
        return Confidence.HIGH

    # HIGH: strong law match alone is enough
    if law_strong:
        return Confidence.HIGH

    # MEDIUM: reasonable law match
    if law_weak:
        return Confidence.MEDIUM

    # MEDIUM: no law but lease itself answers the question clearly
    if not law_present and lease_present and lease_strong:
        return Confidence.MEDIUM

    # LOW: too weak to trust
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
