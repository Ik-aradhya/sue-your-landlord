from fastapi import APIRouter
from rag.chain import get_key_status

status_router = APIRouter()


@status_router.get("/status/keys")
async def key_status():
    """Return current Groq key rotation status."""
    return get_key_status()
