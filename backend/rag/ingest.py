import re
from typing import Optional

import fitz                        # PyMuPDF
from fastapi import UploadFile
from backend.core.config import settings
from backend.models.schemas import Chunk, DocType
from langchain_text_splitters import RecursiveCharacterTextSplitter

import uuid
from backend.core.database import get_law_collection, get_lease_collection, get_embedding

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # 10MB in bytes


# =============================================================================
# FIX 1 — Extract real section name from each chunk's text
# Previously: one hardcoded section_hint applied to ALL chunks
# Now: each chunk gets its own section extracted from its actual text
# =============================================================================

def _law_act_title(state: Optional[str]) -> str:
    s = (state or "").lower()
    if s == "maharashtra":
        return "Maharashtra Rent Control Act, 1999"
    if s == "gujarat":
        return "Gujarat Rent Control Act, 1999"
    return "Rent Control Act"


def _normalize_statute_text(t: str) -> str:
    """Collapse odd PDF whitespace so section headings match reliably."""
    if not t:
        return ""
    t = t.replace("\u00a0", " ").replace("\u2009", " ").replace("\u2007", " ").replace("\u202f", " ")
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return t


# Patterns return group(1) = section number (Arabic digits, optional letter suffix)
_LAW_SECTION_NUMBER_PATTERNS: tuple[str, ...] = (
    r"(?i)\bSection\s+(\d+[A-Z]?)\b",
    r"(?i)\bSections\s+(\d+[A-Z]?)\b",
    r"(?i)\bSec\.?\s*(\d+[A-Z]?)\b",
    r"(?i)§\s*(\d+[A-Z]?)\b",
    r"(?i)\bS\.\s*(\d+[A-Z]?)\b",
    r"(?i)\bunder\s+section\s+(\d+[A-Z]?)\b",
    r"(?i)\bsection\s*[:\-–]\s*(\d+[A-Z]?)\b",
)


def find_last_law_section_citation_in_text(text: str, state: Optional[str]) -> Optional[str]:
    """
    Rightmost section marker in `text` wins (handles multi-section chunks).
    Returns '…Act…, Section N' or None.
    """
    if not text or not text.strip():
        return None
    act = _law_act_title(state)
    norm = _normalize_statute_text(text)
    best_end = -1
    best_num: Optional[str] = None
    for pat in _LAW_SECTION_NUMBER_PATTERNS:
        for m in re.finditer(pat, norm):
            if m.end() > best_end:
                best_end = m.end()
                best_num = m.group(1)
    if not best_num:
        return None
    return f"{act}, Section {best_num}"


def find_law_section_citation(text: str, state: Optional[str]) -> Optional[str]:
    """Backward-compatible name: last section citation in this string."""
    return find_last_law_section_citation_in_text(text, state)


def extract_section_from_chunk(text: str, doc_type: DocType, state: str = None) -> str:
    """
    Extract the real section/clause name from a chunk's text content.
    Called per-chunk so each chunk gets a meaningful, unique citation.
    """

    if doc_type == DocType.LEASE:
        # Match: "Clause 3", "CLAUSE 3A", "Article 5", "ARTICLE 5B"
        match = re.search(r'(?:Clause|CLAUSE|Article|ARTICLE)\s*(\d+[A-Z]?)', text)
        if match:
            # Try to grab the clause title on the same line
            title_match = re.search(
                r'(?:Clause|CLAUSE|Article|ARTICLE)\s*\d+[A-Z]?\s*[:\-–]?\s*([^\n]{5,60})', text
            )
            title = title_match.group(1).strip().rstrip('.,') if title_match else ""
            return f"Clause {match.group(1)}" + (f" - {title}" if title else "")
        return "Lease Document"

    if doc_type == DocType.LAW:
        found = find_last_law_section_citation_in_text(text, state)
        if found:
            return found
        return _law_act_title(state)

    return "Unknown Section"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_pdf(file: UploadFile) -> dict:
    """
    Validates an uploaded PDF before processing.
    Returns: { "is_valid": bool, "error_type": str | None }
    Guarantee: if is_valid=True, file is safe to extract text from.
    """

    # Check 1 — file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_BYTES:
        return {"is_valid": False, "error_type": "FILE_TOO_LARGE"}

    # Check 2 — MIME type
    if file.content_type != "application/pdf":
        return {"is_valid": False, "error_type": "INVALID_FILE_TYPE"}

    # Check 3 — contains extractable text (not a scanned image)
    try:
        raw = file.file.read()
        file.file.seek(0)
        doc = fitz.open(stream=raw, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if len(text.strip()) < 50:
            return {"is_valid": False, "error_type": "NON_TEXT_PDF"}

    except Exception:
        return {"is_valid": False, "error_type": "PARSE_ERROR"}

    return {"is_valid": True, "error_type": None}


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text(file: UploadFile, doc_type: DocType) -> dict:
    """
    Extracts clean text from a validated PDF.

    Input:  validated UploadFile, doc_type (law or lease)
    Output: { text, page_count, doc_type, error_type }

    Guarantee: if error_type is None → text is safe to chunk.
    """
    try:
        raw = file.file.read()
        file.file.seek(0)

        doc = fitz.open(stream=raw, filetype="pdf")
        page_count = len(doc)
        full_text = ""

        for page_num, page in enumerate(doc):
            page_text = page.get_text()

            if len(page_text.strip()) < 20:
                continue

            full_text += f"\n[Page {page_num + 1}]\n"
            full_text += page_text

        doc.close()

        if len(full_text.strip()) < 50:
            return {
                "text": None,
                "page_count": page_count,
                "doc_type": doc_type,
                "error_type": "EMPTY_TEXT"
            }

        return {
            "text": full_text.strip(),
            "page_count": page_count,
            "doc_type": doc_type,
            "error_type": None
        }

    except Exception as e:
        return {
            "text": None,
            "page_count": 0,
            "doc_type": doc_type,
            "error_type": "PARSE_ERROR"
        }


# =============================================================================
# CHUNKING
# =============================================================================

# How far back (in chars) to search for a section heading when the chunk body
# does not contain "Section N" — fixes splits where the heading sits in prior text.
LAW_SECTION_LOOKBACK_CHARS = 15_000

CHUNK_CONFIG = {
    DocType.LAW: {
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    DocType.LEASE: {
        "chunk_size": 256,
        "chunk_overlap": 30
    }
}


def chunk_text(
    text: str,
    doc_type: DocType,
    state: str = None,
    session_id: str = None
    # FIX 2 — removed section_hint parameter entirely
    # Each chunk now extracts its own section via extract_section_from_chunk()
) -> dict:
    """
    Splits extracted text into overlapping chunks with metadata.

    Input:  clean text string, doc_type, optional state/session_id
    Output: { chunks: list[Chunk], error_type }

    Guarantee: every chunk carries source metadata for citation.
    Each chunk gets its own section extracted from its actual text content.
    """

    if not text or len(text.strip()) == 0:
        return {"chunks": [], "error_type": "EMPTY_INPUT"}

    config = CHUNK_CONFIG[doc_type]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", ".", " "],
        length_function=len
    )

    raw_chunks = splitter.split_text(text)

    if len(raw_chunks) == 0:
        return {"chunks": [], "error_type": "NO_CHUNKS_CREATED"}

    chunks = []
    position = 0
    # Law PDFs often put "Section N" in a heading chunk; body chunks need the same label.
    last_law_section: Optional[str] = None

    for i, chunk_text_content in enumerate(raw_chunks):
        start = text.find(chunk_text_content, position)
        end = start + len(chunk_text_content)
        position = max(0, end - config["chunk_overlap"])

        if doc_type == DocType.LAW:
            window = text[max(0, end - LAW_SECTION_LOOKBACK_CHARS) : end]
            from_chunk = find_last_law_section_citation_in_text(chunk_text_content, state)
            from_window = find_last_law_section_citation_in_text(window, state)
            if from_chunk:
                last_law_section = from_chunk
                section = from_chunk
            elif from_window:
                last_law_section = from_window
                section = from_window
            elif last_law_section:
                section = last_law_section
            else:
                section = _law_act_title(state)
        else:
            section = extract_section_from_chunk(chunk_text_content, doc_type, state)

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text_content,
            source=doc_type,
            section=section,        # ✅ unique and meaningful per chunk
            state=state,
            session_id=session_id,
            start_index=start,
            end_index=end
        )
        chunks.append(chunk)

    return {"chunks": chunks, "error_type": None}


# =============================================================================
# FULL INGESTION PIPELINE
# =============================================================================

async def run_ingestion_pipeline(
    file: UploadFile,
    doc_type: DocType,
    state: str = None,
    session_id: str = None
) -> dict:
    """
    Full ingestion pipeline:
    1. Validate PDF
    2. Extract text
    3. Chunk text (each chunk extracts its own section)
    4. Create embeddings
    5. Store in correct collection (law or lease) with full metadata

    Returns:
        {
            "success": bool,
            "session_id": str | None,
            "chunks_stored": int,
            "error_type": str | None
        }
    """

    # STEP 1 — VALIDATE
    validation = validate_pdf(file)
    if not validation["is_valid"]:
        return {
            "success": False,
            "session_id": None,
            "chunks_stored": 0,
            "error_type": validation["error_type"]
        }

    # STEP 2 — EXTRACT TEXT
    extracted = extract_text(file, doc_type)
    if extracted["error_type"]:
        return {
            "success": False,
            "session_id": None,
            "chunks_stored": 0,
            "error_type": extracted["error_type"]
        }

    # STEP 3 — CHUNK TEXT
    # FIX 2 — no section_hint passed; each chunk extracts its own section
    chunked = chunk_text(
        text=extracted["text"],
        doc_type=doc_type,
        state=state,
        session_id=session_id
    )

    if chunked["error_type"]:
        return {
            "success": False,
            "session_id": None,
            "chunks_stored": 0,
            "error_type": chunked["error_type"]
        }

    chunks = chunked["chunks"]

    # STEP 4 — STORE EMBEDDINGS
    try:
        # FIX 1 (original) — correct collection based on doc_type
        if doc_type == DocType.LAW:
            collection = get_law_collection()
        else:
            collection = get_lease_collection()

        documents = [chunk.text for chunk in chunks]
        embeddings = [get_embedding(doc) for doc in documents]
        ids = [chunk.chunk_id for chunk in chunks]

        # Store full metadata — session_id is critical for lease retrieval
        metadatas = [{
            "source":      chunk.source.value,
            "section":     chunk.section or "",     # ✅ now has real section name
            "state":       chunk.state or "",
            "session_id":  chunk.session_id or "",  # ✅ required for lease retrieval
            "start_index": chunk.start_index,
            "end_index":   chunk.end_index
        } for chunk in chunks]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    except Exception as e:
        return {
            "success": False,
            "session_id": None,
            "chunks_stored": 0,
            "error_type": f"VECTOR_DB_ERROR: {str(e)}"
        }

    # SUCCESS
    return {
        "success": True,
        "session_id": session_id,
        "chunks_stored": len(chunks),
        "pages": extracted["page_count"],
        "error_type": None
    }