"""Feature engineering pipeline for credit risk scoring.

Transforms raw loan application columns into 26 ML-ready features including:
ratio features, log transforms, interaction terms, and ordinal buckets.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "loan_amount",
    "annual_income",
    "installment",
    "interest_rate",
    "loan_term_months",
    "fico_score",
    "revolving_utilization",
    "revolving_balance",
    "delinquencies_2y",
    "credit_history_months",
    "open_accounts",
    "total_accounts",
    "public_records",
    "debt_to_income",
    "installment_to_income",
    "loan_to_income_ratio",
    "credit_util_x_delinq",
    "income_x_credit_history",
    "log_annual_income",
    "log_loan_amount",
    "log_revolving_balance",
    "income_bucket",
    "credit_score_bucket",
    "rate_squared",
    "rate_x_term",
    "income_z",
]

_INCOME_BINS = [0, 30_000, 60_000, 100_000, 200_000, float("inf")]
_FICO_BINS = [300, 580, 670, 740, 800, 850]
_BIN_LABELS = [0, 1, 2, 3, 4]


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforms raw loan application data into ML-ready credit features.

    Fit computes income statistics for z-score normalisation. Transform is
    deterministic given the same fit parameters.
    """

    def __init__(self) -> None:
        self._income_mean: float = 0.0
        self._income_std: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CreditFeatureEngineer":
        """Compute training-set income statistics for z-score."""
        self._income_mean = float(X["annual_income"].mean())
        self._income_std = float(X["annual_income"].std()) + 1e-8
        logger.debug(
            "CreditFeatureEngineer fit income_mean=%.2f income_std=%.2f",
            self._income_mean,
            self._income_std,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations and return enriched DataFrame."""
        df = X.copy()

        # Ratio features
        df["debt_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
        df["installment_to_income"] = df["installment"] / (df["annual_income"] / 12 + 1)
        df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1)

        # Interaction features
        df["credit_util_x_delinq"] = df["revolving_utilization"] * (df["delinquencies_2y"] + 1)
        df["income_x_credit_history"] = df["annual_income"] * np.log1p(df["credit_history_months"])

        # Log transforms for skewed distributions
        df["log_annual_income"] = np.log1p(df["annual_income"])
        df["log_loan_amount"] = np.log1p(df["loan_amount"])
        df["log_revolving_balance"] = np.log1p(df["revolving_balance"])

        # Ordinal bucket encodings
        df["income_bucket"] = pd.cut(
            df["annual_income"],
            bins=_INCOME_BINS,
            labels=_BIN_LABELS,
        ).astype(float)

        df["credit_score_bucket"] = pd.cut(
            df["fico_score"],
            bins=_FICO_BINS,
            labels=_BIN_LABELS,
        ).astype(float)

        # Polynomial and interaction rate features
        df["rate_squared"] = df["interest_rate"] ** 2
        df["rate_x_term"] = df["interest_rate"] * df["loan_term_months"]

        # Z-score normalised income
        df["income_z"] = (df["annual_income"] - self._income_mean) / self._income_std

        return df.fillna(0)


def build_feature_pipeline() -> Pipeline:
    """Return a sklearn Pipeline combining feature engineering and scaling."""
    return Pipeline(
        [
            ("engineer", CreditFeatureEngineer()),
            ("scaler", StandardScaler()),
        ]
    )
