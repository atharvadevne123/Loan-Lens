"""Fairness metrics tests."""

import numpy as np
import pandas as pd
import pytest

from app.fairness import (
    demographic_parity_difference,
    equalized_odds_difference,
    compute_fairness_report,
)


def test_demographic_parity_zero():
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dpd = demographic_parity_difference(y_pred, sensitive)
    assert dpd == 0.0


def test_demographic_parity_nonzero():
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dpd = demographic_parity_difference(y_pred, sensitive)
    assert dpd == 1.0


def test_demographic_parity_wrong_groups():
    with pytest.raises(ValueError):
        demographic_parity_difference(np.array([0, 1, 2]), np.array([0, 1, 2]))


def test_equalized_odds_structure():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    sensitive = rng.integers(0, 2, n)
    result = equalized_odds_difference(y_true, y_pred, sensitive)
    assert "tpr_gap" in result
    assert "fpr_gap" in result
    assert result["tpr_gap"] >= 0
    assert result["fpr_gap"] >= 0


def test_compute_fairness_report(synthetic_dataframe):
    X, y_true = synthetic_dataframe
    y_pred = np.random.randint(0, 2, len(y_true))
    report = compute_fairness_report(X, y_true.values, y_pred, sensitive_col="fico_score")
    assert "demographic_parity_difference" in report
    assert "tpr_gap" in report


def test_compute_fairness_report_missing_col(synthetic_dataframe):
    X, y_true = synthetic_dataframe
    y_pred = np.zeros(len(y_true), dtype=int)
    report = compute_fairness_report(X, y_true.values, y_pred, sensitive_col="nonexistent_col")
    assert "error" in report
