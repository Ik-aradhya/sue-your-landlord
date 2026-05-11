# tests/debug_retrieval.py
import sys
sys.path.append(".")

from backend.core.database import get_law_collection, get_embedding

collection = get_law_collection()
query = get_embedding("Can landlord increase rent during active lease?")

results = collection.query(
    query_embeddings=[query],
    where={"state": "maharashtra"},
    n_results=5
)

for i, doc in enumerate(results["documents"][0]):
    distance = results["distances"][0][i]
    meta = results["metadatas"][0][i]
    print(f"\n--- Chunk {i+1} (distance: {distance:.4f}) ---")
    print(f"Section: {meta.get('section')}")
    print(f"Text: {doc[:300]}")

    