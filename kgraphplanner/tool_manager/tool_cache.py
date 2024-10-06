from functools import lru_cache
from typing import Any, Dict

class ToolCache:
    def __init__(self):
        # Store cached values in a dictionary to allow individual key clearing
        self._cache_store: Dict[str, Any] = {}
        self._cache = lru_cache(maxsize=1024)(self._cache_function)

    def _cache_function(self, key: str) -> Any:
        return self._cache_store.get(key, f"Cached result for {key}")

    def get_from_cache(self, key: str) -> Any:
        # Retrieve a value from the cache
        return self._cache(key)

    def put_in_cache(self, key: str, value: Any) -> None:
        # Add a value to the cache by updating the cache store
        self._cache_store[key] = value
        # Update the LRU cache
        self._cache(key)

    def clear_cache(self) -> None:
        # Clear the entire cache
        self._cache.cache_clear()
        self._cache_store.clear()

    def clear_key(self, key: str) -> None:
        # Clear a specific key from the cache
        if key in self._cache_store:
            del self._cache_store[key]
        self._cache.cache_clear()
        # Rebuild the cache without the cleared key
        for k in self._cache_store:
            self._cache(k)