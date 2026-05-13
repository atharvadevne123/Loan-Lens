"""Tests for batch prediction."""

import pytest

from app.batch import MAX_BATCH_SIZE, batch_predict


@pytest.fixture
def applications(sample_application):
    return [dict(sample_application) for _ in range(5)]


def test_batch_predict_returns_correct_count(applications):
    results = batch_predict(applications)
    assert len(results) == 5


def test_batch_predict_structure(applications):
    results = batch_predict(applications)
    for r in results:
        assert "probability" in r
        assert "prediction" in r
        assert "risk_level" in r


def test_batch_predict_single(sample_application):
    results = batch_predict([sample_application])
    assert len(results) == 1
    assert 0.0 <= results[0]["probability"] <= 1.0


def test_batch_predict_exceeds_limit(sample_application):
    with pytest.raises(ValueError, match="exceeds limit"):
        batch_predict([sample_application] * (MAX_BATCH_SIZE + 1))


def test_batch_predict_preserves_order(sample_application):
    app_high = dict(sample_application)
    app_high["fico_score"] = 850
    app_low = dict(sample_application)
    app_low["fico_score"] = 580
    results = batch_predict([app_high, app_low, app_high])
    assert len(results) == 3
    # All should have valid probabilities
    for r in results:
        assert r["probability"] is not None
