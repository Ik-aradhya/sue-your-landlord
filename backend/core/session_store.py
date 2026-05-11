# core/session_store.py

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
import threading

class SessionStore:
    """
    Thread-safe in-memory session store.
    Swap _store backend for Redis/Postgres in production.
    """

    def __init__(self, ttl_minutes: int = 60):
        self._store: dict[str, list] = defaultdict(list)
        self._timestamps: dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(minutes=ttl_minutes)

    def get_history(self, session_id: str) -> list[dict]:
        """Return conversation history for session, [] if expired/missing."""
        with self._lock:
            if self._is_expired(session_id):
                self._evict(session_id)
                return []
            return list(self._store[session_id])

    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        metadata: Optional[dict] = None
    ):
        """Append a Q&A turn to session history."""
        with self._lock:
            self._store[session_id].append({
                "role_user":      question,
                "role_assistant": answer,
                "timestamp":      datetime.utcnow().isoformat(),
                "metadata":       metadata or {}
            })
            self._timestamps[session_id] = datetime.utcnow()

    def clear_session(self, session_id: str):
        with self._lock:
            self._evict(session_id)

    def _is_expired(self, session_id: str) -> bool:
        ts = self._timestamps.get(session_id)
        if not ts:
            return False
        return datetime.utcnow() - ts > self._ttl

    def _evict(self, session_id: str):
        self._store.pop(session_id, None)
        self._timestamps.pop(session_id, None)


# Singleton — import this everywhere
session_store = SessionStore(ttl_minutes=60)