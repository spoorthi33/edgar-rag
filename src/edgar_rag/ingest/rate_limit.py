"""Request throttling for the SEC's 10 requests/second per-IP cap.

Exceeding the cap returns 429 and can get the IP temporarily blocked, so
this is enforced on our side rather than discovered on theirs.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Blocks callers so requests are spaced at least `1/rate` apart."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> float:
        """Wait until the next request is permitted. Returns seconds slept."""
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if wait > 0:
            time.sleep(wait)
        return wait
