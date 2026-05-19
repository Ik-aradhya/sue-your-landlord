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
        if "rate_limit" in error_msg.lower():
            return {"llm_output": None, "error_type": "RATE_LIMIT"}
        return {"llm_output": None, "error_type": f"LLM_ERROR: {error_msg}"}


def format_response(
    llm_output: str,
    confidence: Confidence
) -> RAGResponse:
    """
    Parses raw LLM output into structured RAGResponse.

    Confidence comes from your retrieval system — NOT from the LLM.
    The LLM does not get to decide how confident it is.
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
    ]

    answer_lower = answer.lower()
    hedged = any(phrase in answer_lower for phrase in HEDGE_PHRASES)

    final_confidence = llm_confidence or confidence

    if final_confidence == Confidence.HIGH and not (legal_basis or lease_ref):
        final_confidence = Confidence.MEDIUM

    if final_confidence == Confidence.HIGH and (hedged or conflict_flag):
        final_confidence = Confidence.MEDIUM

    if final_confidence in (Confidence.HIGH, Confidence.MEDIUM) and hedged and conflict_flag:
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


def fallback_response(reason: str = "") -> RAGResponse:
    """
    Returns a safe, honest response when confidence is LOW
    or any pipeline stage fails.
    """
    return RAGResponse(
        answer="I don't have enough information in the provided documents to answer this confidently.",
        legal_basis="",
        lease_reference=None,
        explanation="The retrieved context was insufficient to provide a reliable legal answer. Please consult a legal professional.",
        confidence=Confidence.LOW,
        conflict_flag=False,
        error_type=f"FALLBACK: {reason}" if reason else "FALLBACK"
    )


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
        return fallback_response(llm_result["error_type"])

    # Stage 5 — Format and return structured response
    parsed = format_response(
        llm_output=llm_result["llm_output"],
        confidence=context.confidence
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
