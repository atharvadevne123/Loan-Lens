"""In-memory feature store for real-time feature serving and caching."""

import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_MAX_CACHE_SIZE = 1000
_DEFAULT_TTL_SECONDS = 300


class FeatureStore:
    """LRU cache for applicant feature vectors with TTL expiry."""

    def __init__(self, max_size: int = _MAX_CACHE_SIZE, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, applicant_id: str) -> dict | None:
        """Return cached features for applicant_id if present and not expired."""
        if applicant_id not in self._cache:
            return None
        features, timestamp = self._cache[applicant_id]
        if time.monotonic() - timestamp > self._ttl:
            del self._cache[applicant_id]
            logger.debug("feature_store_evict applicant_id=%s reason=ttl", applicant_id)
            return None
        self._cache.move_to_end(applicant_id)
        return features

    def set(self, applicant_id: str, features: dict) -> None:
        """Cache feature vector for applicant_id."""
        if applicant_id in self._cache:
            self._cache.move_to_end(applicant_id)
        self._cache[applicant_id] = (features, time.monotonic())
        if len(self._cache) > self._max_size:
            evicted = self._cache.popitem(last=False)
            logger.debug("feature_store_evict applicant_id=%s reason=size", evicted[0])

    def delete(self, applicant_id: str) -> None:
        """Remove an applicant's cached features."""
        self._cache.pop(applicant_id, None)

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


# Module-level singleton
_store = FeatureStore()


def get_feature_store() -> FeatureStore:
    return _store
