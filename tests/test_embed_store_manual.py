# tests/test_embed_store_manual.py
import sys
sys.path.append(".")

import io
from unittest.mock import MagicMock
from backend.rag.ingest import run_ingestion_pipeline
from backend.core.database import get_law_collection
from backend.models.schemas import DocType

def make_upload_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    mock = MagicMock()
    mock.content_type = "application/pdf"
    mock.file = io.BytesIO(raw)
    return mock

def run_test():
    print("\n" + "="*50)
    print("TEST — Full ingestion pipeline")
    print("="*50)

    PDF_PATH = "data/laws/maharashtra_rent_act.pdf"
    file = make_upload_file(PDF_PATH)

    print("\nRunning pipeline...")
    result = run_ingestion_pipeline(
        file=file,
        doc_type=DocType.LAW,
        state="maharashtra"
    )

    print(f"Success       : {result['success']}")
    if not result["success"]:
        print(f"Error         : {result['error_type']}")
        return

    print(f"Pages parsed  : {result['pages']}")
    print(f"Chunks stored : {result['chunks_stored']}")

    # Now verify by querying ChromaDB directly
    print("\n--- Querying ChromaDB to verify storage ---")
    collection = get_law_collection()
    count = collection.count()
    print(f"Total vectors in law_chunks: {count}")

    # Run a real similarity search
    from backend.core.database import get_embedding
    test_question = "Can a landlord increase rent during a lease?"
    query_vector = get_embedding(test_question)

    results = collection.query(
        query_embeddings=[query_vector],
        where={"state": "maharashtra"},
        n_results=3
    )

    print(f"\nTop 3 results for: '{test_question}'")
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        print(f"\nResult {i+1} (distance: {distance:.4f}):")
        print(doc[:200])

    print("\nTEST COMPLETE")

if __name__ == "__main__":
    run_test()