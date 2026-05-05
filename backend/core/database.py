import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client globally
_chroma_client = chromadb.PersistentClient(path="./vectorstore")
_embedding_function = embedding_functions.DefaultEmbeddingFunction()

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for a given text string."""
    return _embedding_function([text])[0]

def get_law_collection():
    """Returns the law chunks collection from ChromaDB."""
    collection = _chroma_client.get_or_create_collection(
        name="law_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    return collection
