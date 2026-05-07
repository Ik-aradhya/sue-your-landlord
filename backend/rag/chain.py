import os
import re
from langchain_groq import ChatGroq
from backend.core.config import settings
from backend.models.schemas import RAGResponse, Confidence
from backend.rag.prompts import build_prompt
from backend.rag.retriever import run_retrieval

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

    answer       = extract_field("ANSWER", llm_output)
    legal_basis  = extract_field("LEGAL BASIS", llm_output)
    lease_ref    = extract_field("LEASE REFERENCE", llm_output)
    explanation  = extract_field("EXPLANATION", llm_output)
    conflict_raw = extract_field("CONFLICT", llm_output)

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

    return RAGResponse(
        answer=answer,
        legal_basis=legal_basis,
        lease_reference=lease_ref if lease_ref else None,
        explanation=explanation,
        confidence=confidence,       # from retrieval — never from LLM
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
    session_id: str
) -> RAGResponse:
    """
    Master function — runs the complete RAG pipeline.

    retrieve → confidence check → prompt → LLM → format

    This is the only function the API route needs to call.
    """

    # Stage 1 — Retrieve relevant chunks
    retrieval = run_retrieval(
        question=question,
        state=state,
        session_id=session_id
    )

    if retrieval["error_type"]:
        return fallback_response(retrieval["error_type"])

    # Stage 2 — Confidence gate
    # If confidence is LOW → stop here, return safe fallback
    if retrieval["should_fallback"]:
        return fallback_response("low retrieval confidence")

    context = retrieval["context"]

    # Stage 3 — Build prompt
    messages = build_prompt(
        question=question,
        law_chunks=context.law_chunks,
        lease_chunks=context.lease_chunks
    )

    # Stage 4 — Call LLM
    llm_result = call_llm(messages)
    if llm_result["error_type"]:
        return fallback_response(llm_result["error_type"])

    # Stage 5 — Format and return structured response
    return format_response(
        llm_output=llm_result["llm_output"],
        confidence=context.confidence
    )