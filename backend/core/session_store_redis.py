
import redis
import json

class RedisSessionStore:
    def __init__(self, url: str, ttl_seconds: int = 3600):
        self._client = redis.from_url(url)
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"session:history:{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        raw = self._client.get(self._key(session_id))
        return json.loads(raw) if raw else []

    def append_turn(self, session_id: str, question: str,
                    answer: str, metadata: dict = None):
        history = self.get_history(session_id)
        history.append({
            "role_user":      question,
            "role_assistant": answer,
            "metadata":       metadata or {}
        })
        self._client.setex(
            self._key(session_id),
            self._ttl,
            json.dumps(history)
        )

    def clear_session(self, session_id: str):
        self._client.delete(self._key(session_id))