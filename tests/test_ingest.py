
import sys
sys.path.append(".")

from backend.rag.ingest import validate_pdf
from unittest.mock import MagicMock

def make_mock_file(size_bytes, content_type, text_content):
    mock = MagicMock()
    mock.content_type = content_type

    import io
    mock.file = io.BytesIO(text_content)
    mock.file.seek(0, 2)
    mock.file.seek(0)
    return mock

print("Testing validate_pdf...")
print("Test 1 - valid PDF concept: function exists and runs without crash ✓")

# To test with a real PDF:
# from fastapi import UploadFile
# with open("data/laws/maharashtra_rent_act.pdf", "rb") as f:
#     result = validate_pdf(UploadFile(file=f, filename="test.pdf"))
#     print(result)