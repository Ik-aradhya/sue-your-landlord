import sys
sys.path.append(".")
from backend.core.database import get_lease_collection

collection = get_lease_collection()
print(f"Total lease chunks: {collection.count()}")
docs = collection.get()
for i, meta in enumerate(docs.get("metadatas", [])[:5]):
    print(f"Chunk {i}: session_id={meta.get('session_id')}, section={meta.get('section')}")
