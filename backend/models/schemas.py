from pydantic import BaseModel
from typing import Optional
from enum import Enum

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