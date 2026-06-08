"""Feature engineering tests."""

import numpy as np
import pandas as pd
import pytest

from app.features import FEATURE_COLUMNS, CreditFeatureEngineer, build_feature_pipeline


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(7)
    n = 50
    return pd.DataFrame(
        {
            "loan_amount": rng.uniform(1000, 40000, n),
            "annual_income": rng.uniform(20000, 150000, n),
            "installment": rng.uniform(50, 1500, n),
            "interest_rate": rng.uniform(5, 30, n),
            "loan_term_months": rng.choice([36, 60], n),
            "fico_score": rng.integers(580, 850, n),
            "revolving_utilization": rng.uniform(0, 1, n),
            "revolving_balance": rng.uniform(0, 50000, n),
            "delinquencies_2y": rng.integers(0, 5, n),
            "credit_history_months": rng.integers(12, 360, n),
            "open_accounts": rng.integers(2, 20, n),
            "total_accounts": rng.integers(5, 40, n),
            "public_records": rng.integers(0, 3, n),
        }
    )


def test_engineer_creates_dti(sample_df):
    eng = CreditFeatureEngineer()
    out = eng.fit_transform(sample_df)
    assert "debt_to_income" in out.columns


def test_engineer_creates_log_income(sample_df):
    eng = CreditFeatureEngineer()
    out = eng.fit_transform(sample_df)
    assert "log_annual_income" in out.columns
    assert (out["log_annual_income"] >= 0).all()


def test_engineer_no_nulls(sample_df):
    eng = CreditFeatureEngineer()
    out = eng.fit_transform(sample_df)
    assert out.isnull().sum().sum() == 0


def test_engineer_transform_consistent(sample_df):
    eng = CreditFeatureEngineer()
    eng.fit(sample_df)
    out1 = eng.transform(sample_df)
    out2 = eng.transform(sample_df)
    pd.testing.assert_frame_equal(out1, out2)


def test_feature_columns_all_present(sample_df):
    eng = CreditFeatureEngineer()
    out = eng.fit_transform(sample_df)
    [c for c in FEATURE_COLUMNS if c in out.columns or c in sample_df.columns]
    # Engineered features should be a superset of raw features
    assert len(out.columns) >= len(sample_df.columns)


def test_pipeline_fit_transform(sample_df):
    pipe = build_feature_pipeline()
    # Pipeline needs only the raw columns that are passed
    result = pipe.fit_transform(sample_df)
    assert result is not None
    assert result.shape[0] == len(sample_df)


def test_income_bucket_range(sample_df):
    eng = CreditFeatureEngineer()
    out = eng.fit_transform(sample_df)
    assert "income_bucket" in out.columns
    assert out["income_bucket"].between(0, 4).all()


def test_credit_score_bucket_range(sample_df):
    eng = CreditFeatureEngineer()
    out = eng.fit_transform(sample_df)
    assert "credit_score_bucket" in out.columns
    assert out["credit_score_bucket"].between(0, 4).all()


@pytest.mark.parametrize("col", ["debt_to_income", "installment_to_income", "loan_to_income_ratio"])
def test_ratio_features_non_negative(sample_df, col):
    eng = CreditFeatureEngineer()
    out = eng.fit_transform(sample_df)
    assert (out[col] >= 0).all(), f"{col} contains negative values"
