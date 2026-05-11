# test_embed_store_manual.py — fixed
import sys
sys.path.append(".")

import io
import asyncio
from unittest.mock import MagicMock
from backend.rag.ingest import run_ingestion_pipeline
from backend.rag.chain import run_rag_chain
from backend.models.schemas import DocType
from backend.core.database import get_lease_collection
import uuid

def make_upload_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    mock = MagicMock()
    mock.content_type = "application/pdf"
    mock.file = io.BytesIO(raw)
    return mock

async def run_test():
    print("\n" + "="*50)
    print("TEST — Full lease pipeline")
    print("="*50)

    # ✅ Use a real UUID like the route does
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    LEASE_PATH = "data/uploads/priya_lease.pdf"  # ← your lease PDF
    file = make_upload_file(LEASE_PATH)

    # Step 1 — Ingest lease (not law!)
    print("\nIngesting lease...")
    result = await run_ingestion_pipeline(
        file=file,
        doc_type=DocType.LEASE,          # ← LEASE not LAW
        state="maharashtra",
        session_id=session_id            # ← real UUID
    )

    print(f"Success       : {result['success']}")
    if not result["success"]:
        print(f"Error         : {result['error_type']}")
        return

    print(f"Chunks stored : {result['chunks_stored']}")

    # Step 2 — Verify chunks are in ChromaDB under correct session_id
    print("\n--- Verifying ChromaDB ---")
    collection = get_lease_collection()
    stored = collection.get(
        where={"session_id": session_id},
        include=["documents", "metadatas"]
    )
    print(f"Chunks found for session: {len(stored['documents'])}")
    if stored["documents"]:
        print(f"Sample chunk : {stored['documents'][0][:200]}")
        print(f"Metadata     : {stored['metadatas'][0]}")

    # Step 3 — Run full RAG chain with same session_id
    print("\n--- Running RAG chain ---")
    response = run_rag_chain(
        question="Can the landlord increase rent during the lease?",
        state="maharashtra",
        session_id=session_id            # ← same UUID passed to chat
    )

    print(f"Answer      : {response.answer}")
    print(f"Confidence  : {response.confidence}")
    print(f"Lease ref   : {response.lease_reference}")

if __name__ == "__main__":
    asyncio.run(run_test())