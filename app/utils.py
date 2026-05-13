"""Shared utility functions for Loan-Lens."""

import hashlib
import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def now_utc() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def stable_hash(data: dict) -> str:
    """Return a deterministic SHA-256 hash of a JSON-serialisable dict."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, value))


def format_probability(p: float, decimals: int = 4) -> float:
    """Round and clamp a probability to [0, 1]."""
    return round(clamp(p, 0.0, 1.0), decimals)


def risk_tier(probability: float) -> str:
    """Map a default probability to a risk tier label."""
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"
