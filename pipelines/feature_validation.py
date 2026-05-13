"""Data quality validation for incoming loan application features."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_RANGES = {
    "loan_amount": (0, 500_000),
    "annual_income": (0, 10_000_000),
    "installment": (0, 10_000),
    "interest_rate": (1, 40),
    "loan_term_months": (12, 84),
    "fico_score": (300, 850),
    "revolving_utilization": (0, 1),
    "revolving_balance": (0, 1_000_000),
    "delinquencies_2y": (0, 100),
    "credit_history_months": (0, 600),
    "open_accounts": (0, 100),
    "total_accounts": (0, 200),
    "public_records": (0, 50),
}


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Check a DataFrame for missing values, type errors, and out-of-range values."""
    issues: list[str] = []

    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            issues.append(f"missing_values col={col} count={count}")

    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col not in df.columns:
            issues.append(f"missing_column col={col}")
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            issues.append(f"out_of_range col={col} count={out_of_range} range=[{lo},{hi}]")

    if df.duplicated().sum() > 0:
        issues.append(f"duplicate_rows count={df.duplicated().sum()}")

    report = {
        "valid": len(issues) == 0,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "issues": issues,
    }
    if issues:
        logger.warning("data_validation_failed issues=%s", issues)
    else:
        logger.info("data_validation_passed n_rows=%d", len(df))
    return report
