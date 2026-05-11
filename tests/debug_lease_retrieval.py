import sys
sys.path.append(".")

from backend.core.database import get_lease_collection, get_embedding

collection = get_lease_collection()
query = get_embedding("Can landlord increase rent during active lease?")

# Let's get all chunks to see them
results = collection.get(include=["documents", "metadatas"])
docs = results["documents"]
print(f"Total lease chunks: {len(docs)}")

# Now let's query the specific question
try:
    query_results = collection.query(
        query_embeddings=[query],
        n_results=5
    )
    for i, doc in enumerate(query_results["documents"][0]):
        distance = query_results["distances"][0][i]
        meta = query_results["metadatas"][0][i]
        print(f"\n--- Chunk {i+1} (distance: {distance:.4f}) ---")
        print(f"Text: {doc[:200]}")
except Exception as e:
    print(f"Error querying: {e}")
