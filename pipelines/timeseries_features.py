"""Time-series feature engineering for credit portfolio analytics."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean, std, min, and max features for a given column.

    Useful for portfolio-level time-series analysis of default rates or
    loan volumes over time.
    """
    if windows is None:
        windows = [7, 14, 30]
    result = df.copy()
    for w in windows:
        result[f"{column}_roll_mean_{w}d"] = result[column].rolling(w, min_periods=1).mean()
        result[f"{column}_roll_std_{w}d"] = result[column].rolling(w, min_periods=1).std().fillna(0)
        result[f"{column}_roll_min_{w}d"] = result[column].rolling(w, min_periods=1).min()
        result[f"{column}_roll_max_{w}d"] = result[column].rolling(w, min_periods=1).max()
    logger.debug("rolling_features_added column=%s windows=%s", column, windows)
    return result


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lag features for capturing temporal autocorrelation."""
    if lags is None:
        lags = [1, 3, 7]
    result = df.copy()
    for lag in lags:
        result[f"{column}_lag_{lag}d"] = result[column].shift(lag).fillna(0)
    return result


def compute_portfolio_default_trend(
    predictions: list[dict],
    date_col: str = "created_at",
) -> pd.DataFrame:
    """Aggregate daily default rate trends from a list of prediction records.

    Returns a DataFrame indexed by date with rolling default rate features.
    """
    df = pd.DataFrame(predictions)
    if date_col not in df.columns or "prediction" not in df.columns:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col])
    daily = df.groupby(df[date_col].dt.date)["prediction"].mean().reset_index()
    daily.columns = ["date", "default_rate"]
    daily = daily.sort_values("date").reset_index(drop=True)
    daily = add_rolling_features(daily, "default_rate", windows=[7, 14, 30])
    daily = add_lag_features(daily, "default_rate", lags=[1, 3, 7])
    logger.info("portfolio_trend_computed n_days=%d", len(daily))
    return daily
