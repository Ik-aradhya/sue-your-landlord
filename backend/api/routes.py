import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from rag.ingest import run_ingestion_pipeline
from rag.chain import run_rag_chain
from core.config import settings
from core.database import cleanup_expired_lease_namespaces, delete_lease_namespace
from core.rate_limiter import rate_limiter
from models.schemas import (
    DocType, ChatRequest, ChatResponse, UploadResponse
)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_lease(
    file: UploadFile = File(...),
    state: str = Form(...)
):
    """
    Receives a lease PDF and state selection.
    Runs the full ingestion pipeline.
    Returns a session_id the client uses for all subsequent chat requests.
    """

    # Validate state selection
    valid_states = ["maharashtra", "gujarat", "delhi", "karnataka", "tamil_nadu",
    "telangana", "haryana", "uttar_pradesh", "west_bengal", "kerala"]
    if state.lower() not in valid_states:
        raise HTTPException(
            status_code=400,
            detail=f"State must be one of: {valid_states}"
        )

    cleanup_expired_lease_namespaces()

    # FIX 3 — session_id generated ONCE here and passed into the pipeline
    session_id = str(uuid.uuid4())

    # FIX 5 — await the async pipeline call, pass session_id in
    result = await run_ingestion_pipeline(
        file=file,
        doc_type=DocType.LEASE,
        state=state.lower(),
        session_id=session_id
    )

    if not result["success"]:
        return UploadResponse(
            success=False,
            error_type=result["error_type"],
            message=_error_message(result["error_type"])
        )

    asyncio.create_task(_delete_lease_namespace_later(session_id))

    return UploadResponse(
        success=True,
        session_id=session_id,          # FIX 3 — return the same session_id to frontend
        chunks_stored=result["chunks_stored"],
        pages=result["pages"],
        ocr_quality=result.get("ocr_quality", "HIGH"),
        error_type=None,
        message=f"Lease uploaded successfully. {result['chunks_stored']} sections indexed."
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, raw_request: Request):
    """
    Receives a question + session_id + state.
    Runs the full RAG chain.
    Returns a structured legal answer with citations.
    """

    # ── Per-IP rate limit ──────────────────────────────
    client_ip = (
        raw_request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or raw_request.client.host
    )
    if not rate_limiter.is_allowed(client_ip):
        retry = rate_limiter.retry_after_seconds(client_ip)
        mins = max(1, retry // 60)
        raise HTTPException(
            status_code=429,
            detail=(
                f"You have reached the limit of {settings.MAX_QUESTIONS_PER_IP} "
                f"questions. Please try again after {mins} minute{'s' if mins != 1 else ''}."
            ),
        )

    # Validate question
    if not request.question or len(request.question.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Question is too short. Please ask a complete question."
        )

    # Validate state
    valid_states = [   "maharashtra", "gujarat", "delhi", "karnataka", "tamil_nadu",
    "telangana", "haryana", "uttar_pradesh", "west_bengal", "kerala"]
    if request.state.lower() not in valid_states:
        raise HTTPException(
            status_code=400,
            detail=f"State must be one of: {valid_states}"
        )

    cleanup_expired_lease_namespaces()

    # Run RAG chain with the lease namespace keyed by this session_id.
    response = run_rag_chain(
        question=request.question.strip(),
        state=request.state.lower(),
        session_id=request.session_id.strip(),
    )

    return ChatResponse(
        answer=response.answer,
        legal_basis=response.legal_basis,
        lease_reference=response.lease_reference,
        explanation=response.explanation,
        confidence=response.confidence.value,
        conflict_flag=response.conflict_flag,
        error_type=response.error_type
    )

async def _delete_lease_namespace_later(session_id: str):
    await asyncio.sleep(settings.LEASE_VECTOR_TTL_MINUTES * 60)
    try:
        delete_lease_namespace(session_id)
    except Exception:
        pass

def _error_message(error_type: str) -> str:
    """
    Converts internal error codes into user-friendly messages.
    Never expose raw error codes to the user.
    """
    messages = {
        "FILE_TOO_LARGE":    "Your file exceeds the 10MB limit. Please upload a smaller file.",
        "INVALID_FILE_TYPE": "Please upload your valid lease as a PDF or photo.",
        "INVALID_LEASE_DOCUMENT": "Please upload your valid lease as a PDF or photo.",
        "NON_TEXT_PDF":      "Your document appears to be scanned. Please upload a text-based document.",
        "PARSE_ERROR":       "We could not read your file. It may be corrupted or OCR processing failed.",
        "EMPTY_TEXT":        "No readable text was found in your file.",
        "NO_CHUNKS_CREATED": "Your document was too short to process.",
        "STORAGE_FAILED":    "We could not store your document. Please try again.",
    }
    return messages.get(error_type, "Something went wrong. Please try again.")
