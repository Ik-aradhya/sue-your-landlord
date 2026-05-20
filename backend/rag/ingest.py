import re
import hashlib
from typing import Optional

import fitz                        # PyMuPDF
from fastapi import UploadFile
import pytesseract
import shutil
import io
from pdf2image import convert_from_bytes
from PIL import Image
from core.config import settings
from models.schemas import Chunk, DocType
from langchain_text_splitters import RecursiveCharacterTextSplitter

import uuid
from core.database import get_law_index, get_embeddings, lease_expires_at

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # 10MB in bytes
OCR_DPI = 200
OCR_MAX_PAGES = 8

tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


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
    if s == "delhi":
        return "Delhi Rent Control Act, 1958"
    if s == "karnataka":
        return "Karnataka Rent Control Act, 2001"
    if s == "tamil_nadu":
        return "Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act, 2017"
    if s == "telangana":
        return "Telangana Rent Control Act, 1950"
    if s == "haryana":
        return "Haryana Urban (Control of Rent and Eviction) Act, 1973"
    if s == "uttar_pradesh":
        return "Uttar Pradesh Urban Buildings (Regulation of Letting, Rent and Eviction) Act, 1972"
    if s == "west_bengal":
        return "West Bengal Premises Tenancy Act, 1997"
    if s == "kerala":
        return "Kerala Buildings (Lease and Rent Control) Act, 1965"
    return "Rent Control Act"


def _normalize_statute_text(t: str) -> str:
    """Collapse odd PDF whitespace so section headings match reliably."""
    if not t:
        return ""
    t = t.replace("\u00a0", " ").replace("\u2009", " ").replace("\u2007", " ").replace("\u202f", " ")
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return t


# Patterns return group(1) = section number (Arabic digits, optional letter suffix).
# These intentionally match headings, not inline cross-references like
# "sub-section (1) of section 14", because those references caused chunks from
# Section 11 to be mislabeled as Section 14.
_LAW_SECTION_HEADING_PATTERNS: tuple[str, ...] = (
    r"(?im)^\s*(\d+[A-Z]?)\.\s+[A-Z][^\n]{3,}",
    r"(?m)^\s*(?:Section|SECTION)\s+(\d+[A-Z]?)\b\s*[:.\-–]",
    r"(?m)^\s*(?:Sec\.|SEC\.)\s*(\d+[A-Z]?)\b\s*[:.\-–]",
    r"(?m)^\s*§\s*(\d+[A-Z]?)\b\s*[:.\-–]",
)


def find_last_law_section_citation_in_text(text: str, state: Optional[str]) -> Optional[str]:
    """
    Rightmost section heading in `text` wins (handles multi-section chunks).
    Returns '…Act…, Section N' or None.
    """
    if not text or not text.strip():
        return None
    act = _law_act_title(state)
    norm = _normalize_statute_text(text)
    best_end = -1
    best_num: Optional[str] = None
    for pat in _LAW_SECTION_HEADING_PATTERNS:
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
        page_match = re.search(r'\[Page\s+(\d+)\]', text, re.IGNORECASE)
        if page_match:
            return f"Lease Document, Page {page_match.group(1)}"
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

ACCEPTED_TYPES = [
    "application/pdf",
    "image/jpeg",
    "image/jpg", 
    "image/png",
    "image/heic"    # iPhone photos
]

def validate_file(file: UploadFile) -> dict:
    """
    Validates an uploaded file before processing.
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
    if file.content_type not in ACCEPTED_TYPES:
        return {"is_valid": False, "error_type": "INVALID_FILE_TYPE"}

    # Check 3 — verify file structure
    try:
        raw = file.file.read()
        file.file.seek(0)
        if file.content_type == "application/pdf":
            doc = fitz.open(stream=raw, filetype="pdf")
            doc.close()
        else:
            img = Image.open(io.BytesIO(raw))
            img.verify()
    except Exception as e:
        print(f"Validation Error: {e}")
        return {"is_valid": False, "error_type": "PARSE_ERROR"}

    return {"is_valid": True, "error_type": None}


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def check_ocr_quality(text: str) -> str:
    """
    Evaluates the quality of extracted text based on the ratio of alphanumeric characters.
    """
    if not text:
        return "LOW"
    
    clean_text = re.sub(r'\s+', '', text)
    if not clean_text:
        return "LOW"
        
    alnum_count = sum(c.isalnum() for c in clean_text)
    ratio = alnum_count / len(clean_text)
    
    if ratio > 0.85:
        return "HIGH"
    elif ratio > 0.70:
        return "MEDIUM"
    else:
        return "LOW"


def extract_text_with_ocr(raw: bytes) -> str:
    """
    Fallback: extract text from scanned/image PDFs using OCR.
    Called when normal PyMuPDF extraction returns empty text.
    """
    try:
        # Keep OCR bounded so scanned PDFs do not exceed Railway's request timeout.
        images = convert_from_bytes(
            raw,
            dpi=OCR_DPI,
            first_page=1,
            last_page=OCR_MAX_PAGES,
            thread_count=1,
        )
        
        full_text = ""
        for page_num, image in enumerate(images):
            # Run OCR on each page image
            page_text = pytesseract.image_to_string(
                image,
                lang="eng"      # add "hin" for Hindi support later
            )
            if page_text.strip():
                full_text += f"\n[Page {page_num + 1}]\n"
                full_text += page_text

        return full_text.strip()

    except Exception as e:
        return ""


def extract_text_from_image(file: UploadFile) -> dict:
    """
    Directly extracts text from uploaded image files.
    """
    try:
        raw = file.file.read()
        file.file.seek(0)
        
        image = Image.open(io.BytesIO(raw))
        
        # Enhance image for better OCR accuracy
        image = image.convert("RGB")
        
        text = pytesseract.image_to_string(image, lang="eng")
        
        if len(text.strip()) < 50:
            return {
                "text": None,
                "page_count": 1,
                "error_type": "EMPTY_TEXT",
                "ocr_quality": None
            }

        return {
            "text": text.strip(),
            "page_count": 1,
            "error_type": None,
            "ocr_quality": check_ocr_quality(text)
        }

    except Exception as e:
        print(f"OCR Extraction Error: {e}")
        return {
            "text": None,
            "page_count": 0,
            "error_type": "PARSE_ERROR",
            "ocr_quality": None
        }


def extract_text(file: UploadFile, doc_type: DocType) -> dict:
    """
    Extracts clean text from a validated file.

    Input:  validated UploadFile, doc_type (law or lease)
    Output: { text, page_count, doc_type, error_type }

    Guarantee: if error_type is None → text is safe to chunk.
    """
    if file.content_type in ACCEPTED_TYPES and file.content_type != "application/pdf":
        result = extract_text_from_image(file)
        result["doc_type"] = doc_type
        return result
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
            # Fallback to OCR
            fallback_text = extract_text_with_ocr(raw)
            if len(fallback_text.strip()) >= 50:
                full_text = fallback_text
            else:
                return {
                    "text": None,
                    "page_count": page_count,
                    "doc_type": doc_type,
                    "error_type": "EMPTY_TEXT",
                    "ocr_quality": None
                }

        ocr_quality = check_ocr_quality(full_text)
        return {
            "text": full_text.strip(),
            "page_count": page_count,
            "doc_type": doc_type,
            "error_type": None,
            "ocr_quality": ocr_quality
        }

    except Exception as e:
        print(f"PyMuPDF Extraction Error: {e}")
        return {
            "text": None,
            "page_count": 0,
            "doc_type": doc_type,
            "error_type": "PARSE_ERROR",
            "ocr_quality": None
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

        if doc_type == DocType.LAW:
            digest = hashlib.sha1(
                f"{state or ''}:{start}:{end}:{chunk_text_content[:120]}".encode("utf-8")
            ).hexdigest()[:16]
            chunk_id = f"law:{state or 'unknown'}:{start}:{end}:{digest}"
        else:
            chunk_id = str(uuid.uuid4())

        chunk = Chunk(
            chunk_id=chunk_id,
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
    validation = validate_file(file)
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
        documents = [chunk.text for chunk in chunks]
        embeddings = get_embeddings(documents, input_type="passage")
        ids = [chunk.chunk_id for chunk in chunks]

        lease_expires = lease_expires_at() if doc_type == DocType.LEASE else None

        # Store full metadata — session_id is critical for lease retrieval
        metadatas = [{
            "source":      chunk.source.value,
            "section":     chunk.section or "",     # ✅ now has real section name
            "state":       chunk.state or "",
            "session_id":  chunk.session_id or "",  # ✅ required for lease retrieval
            "lease_expires_at": lease_expires or 0,
            "start_index": chunk.start_index,
            "end_index":   chunk.end_index
        } for chunk in chunks]

        if doc_type == DocType.LAW:
            law_index = get_law_index()
            vectors = []
            for i in range(len(chunks)):
                meta = metadatas[i].copy()
                meta["text"] = documents[i]
                vectors.append({
                    "id": ids[i],
                    "values": embeddings[i],
                    "metadata": meta
                })
            
            # Pinecone upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                law_index.upsert(vectors=vectors[i:i+batch_size])
        else:
            from core.database import get_pinecone_index
            index = get_pinecone_index()
            vectors = [
                {
                    "id": chunks[i].chunk_id,
                    "values": embeddings[i],
                    "metadata": {
                        **metadatas[i],
                        "text": chunks[i].text
                    }
                }
                for i in range(len(chunks))
            ]
            index.upsert(vectors=vectors, namespace=session_id)

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
        "ocr_quality": extracted.get("ocr_quality", "HIGH"),
        "error_type": None
    }   
