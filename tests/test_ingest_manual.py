# tests/test_ingest_manual.py
import sys
sys.path.append(".")

import fitz
import io
from unittest.mock import MagicMock
from backend.rag.ingest import extract_text, chunk_text
from backend.models.schemas import DocType

def make_upload_file_from_path(path: str):
    """Helper: wraps a real PDF file as a mock UploadFile"""
    with open(path, "rb") as f:
        raw = f.read()

    mock_file = MagicMock()
    mock_file.content_type = "application/pdf"
    mock_file.file = io.BytesIO(raw)
    return mock_file

def run_test():
    print("\n" + "="*50)
    print("MANUAL TEST — extract_text + chunk_text")
    print("="*50)

    # Change this path to your actual law PDF
    PDF_PATH = "data/laws/maharashtra_rent_act.pdf"

    print(f"\nLoading: {PDF_PATH}")
    file = make_upload_file_from_path(PDF_PATH)

    # Test extract_text
    print("\n--- extract_text() ---")
    result = extract_text(file, DocType.LAW)

    if result["error_type"]:
        print(f"FAILED: {result['error_type']}")
        return

    print(f"Pages extracted : {result['page_count']}")
    print(f"Total characters: {len(result['text'])}")
    print(f"First 300 chars :\n{result['text'][:300]}")

    # Test chunk_text
    print("\n--- chunk_text() ---")
    chunk_result = chunk_text(
        text=result["text"],
        doc_type=DocType.LAW,
        state="maharashtra",
        section_hint="Maharashtra Rent Control Act"
    )

    if chunk_result["error_type"]:
        print(f"FAILED: {chunk_result['error_type']}")
        return

    chunks = chunk_result["chunks"]
    print(f"Total chunks created : {len(chunks)}")
    print(f"First chunk ID       : {chunks[0].chunk_id}")
    print(f"First chunk source   : {chunks[0].source}")
    print(f"First chunk state    : {chunks[0].state}")
    print(f"First chunk text     :\n{chunks[0].text[:200]}")
    print(f"\nLast chunk text:\n{chunks[-1].text[:200]}")

    # Verify overlap
    print("\n--- Overlap check ---")
    if len(chunks) >= 2:
        end_of_chunk1 = chunks[0].text[-80:]
        start_of_chunk2 = chunks[1].text[:80]
        print(f"End of chunk 1  : ...{end_of_chunk1}")
        print(f"Start of chunk 2: {start_of_chunk2}...")
        print("(You should see shared words above — that is overlap working)")

    print("\nTEST PASSED")

if __name__ == "__main__":
    run_test()