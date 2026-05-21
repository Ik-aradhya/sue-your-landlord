import os
import re
from typing import Optional
from langchain_groq import ChatGroq
from core.config import settings
from core.session_store import session_store
from models.schemas import RAGResponse, Confidence
from rag.prompts import build_prompt, build_retrieval_queries
from rag.retriever import run_retrieval

# Initialise LLM once at module level
llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.LLM_MODEL,
    temperature=0,
    max_tokens=1000
)

def call_llm(messages: list[dict]) -> dict:
    """
    Sends prompt to Groq and returns raw text response.

    Input:  list of message dicts (system + user)
    Output: { llm_output: str, error_type: str | None }

    Guarantee: does not interpret or format the response.
               only handles communication with the API.
    """
    try:
        response = llm.invoke(messages)
        return {
            "llm_output": response.content,
            "error_type": None
        }
    except Exception as e:
        error_msg = str(e)
        error_lower = error_msg.lower()
        credit_markers = [
            "rate_limit",
            "rate limit",
            "429",
            "too many requests",
            "quota",
            "credit",
            "billing",
            "insufficient",
            "resource exhausted",
        ]
        if any(marker in error_lower for marker in credit_markers):
            return {"llm_output": None, "error_type": "LLM_CREDITS_UNAVAILABLE"}
        return {"llm_output": None, "error_type": f"LLM_ERROR: {error_msg}"}


def format_response(
    llm_output: str,
    confidence: Confidence,
    lease_chunks: Optional[list] = None,
    question: str = "",
) -> RAGResponse:
    """
    Parses raw LLM output into structured RAGResponse.

    Confidence is parsed from the model response, then constrained by
    parser-level quality checks.
    """

    def extract_field(label: str, text: str) -> str:
        """Extracts content after a label like 'ANSWER:' """
        pattern = rf"{label}:\s*(.*?)(?=\n[A-Z ]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def clean_lease_reference(raw: str) -> str:
        """Strips wrapping quotes and collapses extra whitespace from lease reference."""
        if not raw:
            return raw
        cleaned = raw.strip().strip('"').strip("'").strip("“”‘’")
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip().strip(',').strip()
        return cleaned

    def parse_confidence(raw: str) -> Optional[Confidence]:
        value = raw.strip().upper()
        if value.startswith("HIGH"):
            return Confidence.HIGH
        if value.startswith("MEDIUM"):
            return Confidence.MEDIUM
        if value.startswith("LOW"):
            return Confidence.LOW
        return None

    answer       = extract_field("ANSWER", llm_output)
    legal_basis  = extract_field("LEGAL BASIS", llm_output)
    lease_ref    = clean_lease_reference(extract_field("LEASE REFERENCE", llm_output))
    explanation  = extract_field("EXPLANATION", llm_output)
    conflict_raw = extract_field("CONFLICT", llm_output)
    llm_confidence = parse_confidence(extract_field("CONFIDENCE", llm_output))
    lease_ref = format_lease_reference(
        raw_lease_reference=lease_ref,
        lease_chunks=lease_chunks or [],
        question=question,
        supporting_notes=explanation,
    )

    # If parsing failed — return safe fallback
    if not answer:
        return RAGResponse(
            answer="I was unable to parse the response. Please try again.",
            legal_basis="",
            lease_reference=None,
            explanation="",
            confidence=Confidence.LOW,
            conflict_flag=False,
            error_type="PARSING_FAILED"
        )

    conflict_flag = conflict_raw.strip().upper().startswith("YES")

    # Post-process confidence based on LLM output hedging
    HEDGE_PHRASES = [
        "one would need to refer",
        "does not explicitly",
        "may be entitled",
        "might be able to",
        "under certain conditions",
        "it depends",
        "cannot be determined",
        "unclear",
        "ambiguous",
        "possibly",
        "perhaps",
        "could be",
        "could have",
        "may have",
        "may be able",
        "would depend",
        "depends on",
        "not enough information",
        "only if",
        "it depends on your situation",
        "you may have rights",
        "consider seeking legal advice",
        "the context does not provide enough information",
        "consult a legal professional",
        "review the act",
        "seek legal advice",
        "the next step would be to review",
    ]

    combined_lower = f"{answer} {explanation}".lower()
    hedged = any(phrase in combined_lower for phrase in HEDGE_PHRASES)

    final_confidence = llm_confidence or confidence

    if final_confidence == Confidence.HIGH and not (legal_basis or lease_ref):
        final_confidence = Confidence.MEDIUM

    if final_confidence == Confidence.HIGH and hedged:
        final_confidence = Confidence.MEDIUM

    if final_confidence in (Confidence.HIGH, Confidence.MEDIUM) and hedged and not legal_basis:
        final_confidence = Confidence.LOW

    return RAGResponse(
        answer=answer,
        legal_basis=legal_basis,
        lease_reference=lease_ref if lease_ref else None,
        explanation=explanation,
        confidence=final_confidence,
        conflict_flag=conflict_flag,
        error_type=None
    )


def _clean_lease_text(text: str) -> str:
    cleaned = re.sub(r"\[Page\s+\d+\]", " ", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" \"'")


def _issue_topic(question: str) -> str:
    q = (question or "").lower()
    topic_map = [
        (("security deposit", "deposit"), "security deposit"),
        (("rent increase", "increase rent", "rent hike", "escalation", "revision"), "rent increase"),
        (("notice", "vacate", "termination"), "notice or termination"),
        (("receipt", "cash", "payment proof", "paid rent"), "rent payment receipts"),
        (("repair", "maintenance"), "repairs or maintenance"),
        (("lock", "evict", "possession", "leave"), "possession or eviction"),
    ]
    for keywords, topic in topic_map:
        if any(keyword in q for keyword in keywords):
            return topic
    compact = re.sub(r"[^a-z0-9\s]", " ", q)
    words = [
        word for word in compact.split()
        if len(word) > 3 and word not in {
            "what", "when", "where", "which", "landlord", "tenant",
            "lease", "does", "have", "with", "from", "this", "that",
            "your", "mine", "protected", "allowed", "required",
        }
    ]
    return " ".join(words[:3]) or "this issue"


def _question_terms(question: str) -> set[str]:
    topic = _issue_topic(question)
    text = f"{question or ''} {topic}".lower()
    terms = {
        word for word in re.findall(r"[a-z0-9]+", text)
        if len(word) > 3 and word not in {
            "what", "when", "where", "which", "landlord", "tenant",
            "lease", "does", "have", "with", "from", "this", "that",
            "your", "mine", "protected", "allowed", "required",
        }
    }
    if "deposit" in terms:
        terms.update({"security", "refund", "refunded", "deduct", "deduction", "interest"})
    if "rent" in terms or "increase" in terms:
        terms.update({"rent", "increase", "revision", "escalation", "enhance", "enhancement"})
    if "notice" in terms:
        terms.update({"notice", "terminate", "termination", "vacate"})
    return terms


def _lease_heading(chunk, fallback_index: int) -> str:
    section = (getattr(chunk, "section", "") or "").strip()
    if section:
        return section
    return f"Lease Excerpt {fallback_index}"


def _score_lease_chunk(chunk, terms: set[str]) -> int:
    haystack = f"{getattr(chunk, 'section', '')} {getattr(chunk, 'text', '')}".lower()
    return sum(haystack.count(term) for term in terms)


def _select_lease_chunk(raw_reference: str, lease_chunks: list, question: str):
    if not lease_chunks:
        return None, 0

    raw = raw_reference or ""
    clause_match = re.search(r"\b(?:lease\s+)?(?:clause|article)\s+(\d+[a-z]?)\b", raw, re.IGNORECASE)
    if clause_match:
        clause_num = clause_match.group(1)
        clause_pattern = re.compile(rf"\b(?:clause|article)\s+{re.escape(clause_num)}\b", re.IGNORECASE)
        for i, chunk in enumerate(lease_chunks, start=1):
            heading = _lease_heading(chunk, i)
            text_start = _clean_lease_text(getattr(chunk, "text", ""))[:120]
            if clause_pattern.search(f"{heading} {text_start}"):
                return chunk, i
        if clause_num.isdigit():
            synthetic_index = int(clause_num)
            if 1 <= synthetic_index <= len(lease_chunks):
                return lease_chunks[synthetic_index - 1], synthetic_index

    terms = _question_terms(question)
    scored = [
        (_score_lease_chunk(chunk, terms), i, chunk)
        for i, chunk in enumerate(lease_chunks, start=1)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    _, index, chunk = scored[0]
    return chunk, index


def _lease_excerpt(text: str, question: str, max_chars: int = 320) -> str:
    cleaned = _clean_lease_text(text)
    if not cleaned:
        return ""

    terms = _question_terms(question)
    candidates = [
        candidate.strip()
        for candidate in re.split(r"(?<!Rs\.)(?<=[.!?])\s+|\s{2,}", cleaned)
        if len(candidate.strip()) >= 20
    ] or [cleaned]

    def score(candidate: str) -> int:
        lower = candidate.lower()
        return sum(lower.count(term) for term in terms)

    best = max(candidates, key=score)
    if score(best) == 0:
        best = cleaned

    if len(best) <= max_chars:
        return best
    truncated = best[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{truncated}..."


def _lease_reference_says_silent(raw_reference: str) -> bool:
    raw = (raw_reference or "").lower()
    silence_markers = [
        "no clause",
        "no relevant",
        "no direct",
        "silent",
        "contains no provision",
        "no provision",
        "does not address",
        "doesn't address",
        "lacks",
        "not mention",
    ]
    return any(marker in raw for marker in silence_markers)


def format_lease_reference(
    raw_lease_reference: str,
    lease_chunks: list,
    question: str,
    supporting_notes: str = "",
) -> str:
    """
    Render lease references with actual lease text, never clause numbers alone.
    """
    if not lease_chunks:
        return raw_lease_reference

    selected_chunk, index = _select_lease_chunk(raw_lease_reference, lease_chunks, question)
    if not selected_chunk:
        return raw_lease_reference

    heading = _lease_heading(selected_chunk, index)
    excerpt = _lease_excerpt(getattr(selected_chunk, "text", ""), question)
    if not excerpt:
        return raw_lease_reference

    if (
        _lease_reference_says_silent(raw_lease_reference)
        or _lease_reference_says_silent(supporting_notes)
    ):
        topic = _issue_topic(question)
        return (
            f"No clause in this lease directly addresses {topic}. "
            f"Closest relevant clause — {heading}: '{excerpt}'"
        )

    return f"{heading}: '{excerpt}'"


def fallback_response(reason: str = "") -> RAGResponse:
    """
    Returns a safe, honest response when confidence is LOW
    or any pipeline stage fails.
    """
    return RAGResponse(
        answer="No legal provision was retrieved that answers this question.",
        legal_basis="",
        lease_reference=None,
        explanation="Missing: a matching statute section or lease clause for the specific issue. Ask again with the issue and state named clearly, such as repairs, rent increase, lockout, possession, notice, or receipts.",
        confidence=Confidence.LOW,
        conflict_flag=False,
        error_type=f"FALLBACK: {reason}" if reason else "FALLBACK"
    )


def retrieval_unavailable_response(reason: str = "") -> RAGResponse:
    """
    Returns a clear system-status response when Pinecone/embedding retrieval is
    unavailable. This must not look like a legal conclusion.
    """
    return RAGResponse(
        answer="The legal database is temporarily unavailable, so I can't retrieve the law or lease context right now.",
        legal_basis="",
        lease_reference="Retrieval unavailable — lease and law context could not be checked",
        explanation="This is a system retrieval issue, not a legal answer. Please try again after the vector database is available.",
        confidence=Confidence.LOW,
        conflict_flag=False,
        error_type=f"RETRIEVAL_UNAVAILABLE: {reason}" if reason else "RETRIEVAL_UNAVAILABLE"
    )


def llm_unavailable_response(reason: str = "") -> RAGResponse:
    """
    Returns a clear system-status response when the LLM provider cannot answer.
    This must not look like a legal conclusion.
    """
    return RAGResponse(
        answer="AI credits are not available right now. Please try again after some time.",
        legal_basis="",
        lease_reference=None,
        explanation="The language model could not generate an answer because the provider is temporarily unavailable, rate limited, or out of credits.",
        confidence=Confidence.LOW,
        conflict_flag=False,
        error_type=reason or "LLM_UNAVAILABLE"
    )


def is_retrieval_unavailable(reason: str | None) -> bool:
    if not reason:
        return False

    reason_upper = reason.upper()
    unavailable_markers = [
        "EMBEDDING_FAILED",
        "LAW_INDEX_EMPTY_OR_UNAVAILABLE",
        "LAW_RETRIEVAL_FAILED",
        "VECTOR_DB_ERROR",
        "PINECONE",
        "CONNECTION",
        "TIMEOUT",
        "UNAVAILABLE",
        "DNS",
        "SERVICE",
    ]
    return any(marker in reason_upper for marker in unavailable_markers)


def run_rag_chain(
    question: str,
    state: str,
    session_id: str,
    history: Optional[list] = None,
) -> RAGResponse:
    """
    Master function — runs the complete RAG pipeline.

    retrieve → confidence check → prompt → LLM → format

    This is the only function the API route needs to call.
    """

    if history is None:
        history = session_store.get_history(session_id)

    # Rewritten queries are embedded for retrieval; `question` alone is for the LLM.
    retrieval_queries = build_retrieval_queries(question, history)

    # Stage 1 — Retrieve (embedding uses rewritten / history-augmented queries)
    retrieval = run_retrieval(
        question=retrieval_queries[0],
        state=state,
        session_id=session_id,
        retrieval_queries=retrieval_queries,
    )

    if retrieval["error_type"]:
        if is_retrieval_unavailable(retrieval["error_type"]):
            return retrieval_unavailable_response(retrieval["error_type"])
        return fallback_response(retrieval["error_type"])

    context = retrieval["context"]

    # Stage 2 — Confidence gate
    # If retrieval found no usable context, stop. If it found some context, let
    # the prompt decide whether the documents answer the question; this avoids
    # unhelpful fallbacks for valid law questions with weaker vector scores.
    if retrieval["should_fallback"] and not (context.law_chunks or context.lease_chunks):
        return fallback_response("low retrieval confidence")

    # Stage 3 — Build prompt (jurisdiction + conversation history in user message)
    messages = build_prompt(
        question=question,
        law_chunks=context.law_chunks,
        lease_chunks=context.lease_chunks,
        state=state,
        history=history,
    )

    # Stage 4 — Call LLM
    llm_result = call_llm(messages)
    if llm_result["error_type"]:
        return llm_unavailable_response(llm_result["error_type"])

    # Stage 5 — Format and return structured response
    parsed = format_response(
        llm_output=llm_result["llm_output"],
        confidence=context.confidence,
        lease_chunks=context.lease_chunks,
        question=question,
    )

    session_store.append_turn(
        session_id=session_id,
        question=question,
        answer=parsed.answer,
        metadata={
            "confidence": parsed.confidence.value,
            "conflict_flag": parsed.conflict_flag,
            "state": state,
        },
    )

    return parsed
