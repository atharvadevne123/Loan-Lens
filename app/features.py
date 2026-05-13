"""Feature engineering pipeline for credit risk scoring."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforms raw loan application data into ML-ready features."""

    def fit(self, X: pd.DataFrame, y=None) -> "CreditFeatureEngineer":
        self._income_mean = X["annual_income"].mean()
        self._income_std = X["annual_income"].std() + 1e-8
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

        # Binned features
        df["income_bucket"] = pd.cut(
            df["annual_income"],
            bins=[0, 30000, 60000, 100000, 200000, float("inf")],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

        df["credit_score_bucket"] = pd.cut(
            df["fico_score"],
            bins=[300, 580, 670, 740, 800, 850],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

        # Lag / rolling proxies via polynomial of interest rate
        df["rate_squared"] = df["interest_rate"] ** 2
        df["rate_x_term"] = df["interest_rate"] * df["loan_term_months"]

        # Normalised income deviation
        df["income_z"] = (df["annual_income"] - self._income_mean) / self._income_std

        return df.fillna(0)


FEATURE_COLUMNS = [
    "loan_amount", "annual_income", "installment", "interest_rate",
    "loan_term_months", "fico_score", "revolving_utilization", "revolving_balance",
    "delinquencies_2y", "credit_history_months", "open_accounts", "total_accounts",
    "public_records", "debt_to_income", "installment_to_income", "loan_to_income_ratio",
    "credit_util_x_delinq", "income_x_credit_history", "log_annual_income",
    "log_loan_amount", "log_revolving_balance", "income_bucket", "credit_score_bucket",
    "rate_squared", "rate_x_term", "income_z",
]


def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("engineer", CreditFeatureEngineer()),
        ("scaler", StandardScaler()),
    ])
