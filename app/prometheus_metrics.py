"""Prometheus metrics instrumentation for Loan-Lens (no-op if prometheus_client unavailable)."""

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

    PREDICTIONS_TOTAL = Counter(
        "loan_lens_predictions_total",
        "Total predictions served",
        ["risk_level"],
    )
    PREDICTION_LATENCY = Histogram(
        "loan_lens_prediction_latency_seconds",
        "Prediction endpoint latency in seconds",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    )
    DEFAULT_PROBABILITY = Histogram(
        "loan_lens_default_probability",
        "Distribution of predicted default probabilities",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    DRIFT_DETECTED = Gauge(
        "loan_lens_drift_features_total",
        "Number of features with detected drift",
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not installed — metrics instrumentation disabled")


def record_prediction(probability: float, risk_level: str, latency_s: float) -> None:
    """Record a prediction event to Prometheus counters and histograms."""
    if not PROMETHEUS_AVAILABLE:
        return
    PREDICTIONS_TOTAL.labels(risk_level=risk_level).inc()
    PREDICTION_LATENCY.observe(latency_s)
    DEFAULT_PROBABILITY.observe(probability)


def update_drift_gauge(n_drifted: int) -> None:
    """Update the gauge tracking how many features are currently drifted."""
    if not PROMETHEUS_AVAILABLE:
        return
    DRIFT_DETECTED.set(n_drifted)


def get_metrics_text() -> tuple[str, str]:
    """Return Prometheus metrics in text format and the appropriate content type."""
    if not PROMETHEUS_AVAILABLE:
        return "# prometheus_client not installed\n", "text/plain"
    return generate_latest().decode(), CONTENT_TYPE_LATEST
