import fitz                        # PyMuPDF
from fastapi import UploadFile
from backend.core.config import settings
from backend.models.schemas import Chunk, DocType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.database import get_law_collection, get_embedding
import uuid

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # 10MB in bytes

def validate_pdf(file: UploadFile) -> dict:
    """
    Validates an uploaded PDF before processing.
    Returns: { "is_valid": bool, "error_type": str | None }
    Guarantee: if is_valid=True, file is safe to extract text from.
    """

    # Check 1 — file size
    file.file.seek(0, 2)           # seek to end of file
    size = file.file.tell()        # get position = file size in bytes
    file.file.seek(0)              # reset to beginning for later use
    
    if size > MAX_BYTES:
        return {"is_valid": False, "error_type": "FILE_TOO_LARGE"}

    # Check 2 — MIME type
    if file.content_type != "application/pdf":
        return {"is_valid": False, "error_type": "INVALID_FILE_TYPE"}

    # Check 3 — contains extractable text (not a scanned image)
    try:
        raw = file.file.read()
        file.file.seek(0)          # reset again after reading
        doc = fitz.open(stream=raw, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if len(text.strip()) < 50:    # less than 50 chars = likely scanned
            return {"is_valid": False, "error_type": "NON_TEXT_PDF"}

    except Exception:
        return {"is_valid": False, "error_type": "PARSE_ERROR"}

    return {"is_valid": True, "error_type": None}

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

            # Skip pages that are mostly whitespace
            if len(page_text.strip()) < 20:
                continue

            # Label each page so section references survive chunking
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

# Chunk configuration per document type
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
    session_id: str = None,
    section_hint: str = "Unknown Section"
) -> dict:
    """
    Splits extracted text into overlapping chunks with metadata.

    Input:  clean text string, doc_type, optional state/session_id
    Output: { chunks: list[Chunk], error_type }

    Guarantee: every chunk carries source metadata for citation.
    """

    if not text or len(text.strip()) == 0:
        return {"chunks": [], "error_type": "EMPTY_INPUT"}

    config = CHUNK_CONFIG[doc_type]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", ".", " "],  # try paragraph first, then line, then sentence
        length_function=len
    )

    raw_chunks = splitter.split_text(text)

    if len(raw_chunks) == 0:
        return {"chunks": [], "error_type": "NO_CHUNKS_CREATED"}

    chunks = []
    position = 0

    for i, chunk_text_content in enumerate(raw_chunks):
        start = text.find(chunk_text_content, position)
        end = start + len(chunk_text_content)
        position = max(0, end - config["chunk_overlap"])

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text_content,
            source=doc_type,
            section=section_hint,
            state=state,
            session_id=session_id,
            start_index=start,
            end_index=end
        )
        chunks.append(chunk)

    return {"chunks": chunks, "error_type": None}