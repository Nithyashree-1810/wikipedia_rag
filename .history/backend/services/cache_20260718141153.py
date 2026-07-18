# services/cache.py
import time
from typing import Optional

class SimpleCache:
    def __init__(self, ttl_seconds=3600):
        self._store = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[dict]:
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._store[key]
            return None
        return entry["value"]

    def set(self, key: str, value: dict):
        self._store[key] = {"value": value, "ts": time.time()}

    def stats(self) -> dict:
        return {"cached_entries": len(self._store)}

cache = SimpleCache(ttl_seconds=3600)