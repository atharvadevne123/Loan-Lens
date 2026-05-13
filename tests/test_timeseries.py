"""Tests for time-series feature engineering."""

import pandas as pd
import numpy as np
import pytest

from pipelines.timeseries_features import add_rolling_features, add_lag_features, compute_portfolio_default_trend


@pytest.fixture
def daily_df():
    dates = pd.date_range("2024-01-01", periods=30)
    rng = np.random.default_rng(5)
    return pd.DataFrame({"date": dates, "default_rate": rng.uniform(0.1, 0.4, 30)})


def test_add_rolling_features_creates_columns(daily_df):
    result = add_rolling_features(daily_df, "default_rate", windows=[7])
    assert "default_rate_roll_mean_7d" in result.columns
    assert "default_rate_roll_std_7d" in result.columns


def test_add_rolling_features_no_nulls(daily_df):
    result = add_rolling_features(daily_df, "default_rate", windows=[7, 14])
    assert result.isnull().sum().sum() == 0


def test_add_lag_features_creates_columns(daily_df):
    result = add_lag_features(daily_df, "default_rate", lags=[1, 3])
    assert "default_rate_lag_1d" in result.columns
    assert "default_rate_lag_3d" in result.columns


def test_add_lag_features_no_nulls(daily_df):
    result = add_lag_features(daily_df, "default_rate", lags=[1])
    assert result.isnull().sum().sum() == 0


def test_compute_portfolio_trend_empty():
    result = compute_portfolio_default_trend([])
    assert len(result) == 0


def test_compute_portfolio_trend(daily_df):
    records = [
        {"created_at": str(d), "prediction": int(r > 0.25)}
        for d, r in zip(daily_df["date"], daily_df["default_rate"])
    ]
    result = compute_portfolio_default_trend(records)
    assert "default_rate" in result.columns
    assert "default_rate_roll_mean_7d" in result.columns
