"""Drift detection and monitoring tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.monitoring import compute_drift, compute_prediction_stats, log_prediction


def test_compute_drift_no_drift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 200).tolist()
    cur = rng.normal(0, 1, 200).tolist()
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert isinstance(result["drift_detected"], bool)


def test_compute_drift_detects_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500).tolist()
    cur = rng.normal(5, 1, 500).tolist()  # large mean shift
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_same_distribution():
    rng = np.random.default_rng(10)
    data = rng.normal(10, 2, 1000).tolist()
    ref = data[:500]
    cur = data[500:]
    result = compute_drift(ref, cur)
    # With large same-dist samples, drift should usually not be detected
    assert result["p_value"] > 0


def test_log_prediction_persists(db_session):
    record = log_prediction(
        db=db_session,
        correlation_id="test-corr-1",
        input_features={"loan_amount": 10000.0},
        probability=0.3,
        prediction=0,
    )
    assert record.id is not None
    assert record.probability == 0.3
    assert record.prediction == 0


def test_compute_prediction_stats_empty(db_session):
    # Fresh session may have no records
    stats = compute_prediction_stats(db_session, window=0)
    assert "total_predictions" in stats


def test_prediction_stats_after_logging(db_session):
    for i in range(5):
        log_prediction(
            db=db_session,
            correlation_id=f"test-corr-{i}",
            input_features={"loan_amount": float(i * 1000)},
            probability=float(i) / 10,
            prediction=0 if i < 3 else 1,
        )
    stats = compute_prediction_stats(db_session)
    assert stats["total_predictions"] >= 5


@pytest.mark.parametrize("prob,pred", [
    (0.1, 0),
    (0.5, 1),
    (0.9, 1),
])
def test_log_prediction_various_probs(db_session, prob, pred):
    record = log_prediction(
        db=db_session,
        correlation_id=f"param-corr-{prob}",
        input_features={"fico_score": 700},
        probability=prob,
        prediction=pred,
    )
    assert record.probability == prob
    assert record.prediction == pred
