# core/rate_limiter.py

import threading
from collections import defaultdict
from datetime import datetime, timedelta

from core.config import settings


class RateLimiter:
    """
    Thread-safe in-memory per-IP rate limiter.
    Allows up to `max_requests` questions within a rolling `window` period.
    """

    def __init__(
        self,
        max_requests: int = settings.MAX_QUESTIONS_PER_IP,
        window_minutes: int = settings.RATE_LIMIT_WINDOW_MINUTES,
    ):
        self._max = max_requests
        self._window = timedelta(minutes=window_minutes)
        self._hits: dict[str, list[datetime]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        """Return True if the IP has remaining quota, and record the hit."""
        now = datetime.utcnow()
        with self._lock:
            # Evict timestamps outside the window
            self._hits[ip] = [
                ts for ts in self._hits[ip] if now - ts < self._window
            ]
            if len(self._hits[ip]) >= self._max:
                return False
            self._hits[ip].append(now)
            return True

    def remaining(self, ip: str) -> int:
        """Return how many questions the IP can still ask."""
        now = datetime.utcnow()
        with self._lock:
            self._hits[ip] = [
                ts for ts in self._hits[ip] if now - ts < self._window
            ]
            return max(0, self._max - len(self._hits[ip]))

    def retry_after_seconds(self, ip: str) -> int:
        """Seconds until the oldest hit expires and a slot opens."""
        now = datetime.utcnow()
        with self._lock:
            self._hits[ip] = [
                ts for ts in self._hits[ip] if now - ts < self._window
            ]
            if not self._hits[ip]:
                return 0
            oldest = min(self._hits[ip])
            return max(0, int((oldest + self._window - now).total_seconds()))


# Singleton — import this everywhere
rate_limiter = RateLimiter()
