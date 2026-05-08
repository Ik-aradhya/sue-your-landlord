from pydantic import BaseModel
from typing import Optional
from enum import Enum
from typing import List


class DocType(str, Enum):
    LAW = "law"
    LEASE = "lease"

class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Chunk(BaseModel):
    chunk_id: str
    text: str
    source: DocType
    section: str
    state: Optional[str] = None        # law chunks only
    session_id: Optional[str] = None   # lease chunks only
    start_index: int
    end_index: int

class RetrievedContext(BaseModel):
    law_chunks: list[Chunk]
    lease_chunks: list[Chunk]
    law_score: float    # best distance score from law retrieval
    lease_score: float  # best distance score from lease retrieval
    confidence: Confidence

class RAGResponse(BaseModel):
    answer: str
    legal_basis: str
    lease_reference: Optional[str] = None
    explanation: str
    confidence: Confidence
    conflict_flag: bool = False
    error_type: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    chunks_stored: Optional[int] = None
    pages: Optional[int] = None
    error_type: Optional[str] = None
    message:Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    question: str
    state: str                    # "maharashtra" | "gujarat"

class ChatResponse(BaseModel):
    answer: str
    legal_basis: str
    lease_reference: Optional[str] = None
    explanation: str
    confidence: Confidence
    conflict_flag: bool
    error_type: Optional[str] = None