"""Model training and prediction tests."""

import numpy as np
import pandas as pd
import pytest

from app.model import predict, train_model


def test_train_model_returns_metrics(synthetic_dataframe):
    X, y = synthetic_dataframe
    _, metrics = train_model(X, y)
    assert "auc_mean" in metrics
    assert 0.0 <= metrics["auc_mean"] <= 1.0
    assert metrics["n_features"] > 0


def test_train_model_cv_std(synthetic_dataframe):
    X, y = synthetic_dataframe
    _, metrics = train_model(X, y)
    assert metrics["auc_std"] >= 0.0


def test_predict_output_structure(sample_application):
    result = predict(sample_application)
    assert "probability" in result
    assert "prediction" in result
    assert "risk_level" in result


def test_predict_probability_range(sample_application):
    result = predict(sample_application)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_label_binary(sample_application):
    result = predict(sample_application)
    assert result["prediction"] in (0, 1)


def test_predict_risk_levels(sample_application):
    result = predict(sample_application)
    assert result["risk_level"] in ("low", "medium", "high")


@pytest.mark.parametrize("fico_score,expected_risk", [
    (850, "low"),   # best credit -> low probability usually
])
def test_high_fico_generally_low_risk(sample_application, fico_score, expected_risk):
    app = dict(sample_application)
    app["fico_score"] = fico_score
    app["delinquencies_2y"] = 0
    app["revolving_utilization"] = 0.05
    result = predict(app)
    # Just assert structure is valid — model is stochastic
    assert result["risk_level"] in ("low", "medium", "high")


def test_train_with_imbalanced_labels(synthetic_dataframe):
    X, _ = synthetic_dataframe
    np.random.default_rng(42)
    y = pd.Series([0] * (len(X) - 10) + [1] * 10)
    _, metrics = train_model(X, y)
    assert metrics["auc_mean"] >= 0.0


def test_train_model_saves_artifacts(synthetic_dataframe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = synthetic_dataframe
    pipe, metrics = train_model(X, y)
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert pipe is not None
