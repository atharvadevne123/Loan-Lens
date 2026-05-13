"""Fairness metrics for credit risk model evaluation."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Compute demographic parity difference between two groups.

    A value of 0 indicates perfect parity; positive values indicate the
    privileged group receives more positive outcomes.
    """
    unique = np.unique(sensitive)
    if len(unique) != 2:
        raise ValueError(f"Expected 2 groups, got {len(unique)}")
    rates = {g: float(y_pred[sensitive == g].mean()) for g in unique}
    diff = abs(rates[unique[1]] - rates[unique[0]])
    logger.info("demographic_parity group_rates=%s diff=%.4f", rates, diff)
    return round(diff, 4)


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> dict:
    """Compute equalized odds: TPR and FPR gaps across two groups."""
    unique = np.unique(sensitive)
    if len(unique) != 2:
        raise ValueError(f"Expected 2 groups, got {len(unique)}")

    result = {}
    for metric, cond_label in [("tpr", 1), ("fpr", 0)]:
        rates = {}
        for g in unique:
            mask = (sensitive == g) & (y_true == cond_label)
            if mask.sum() == 0:
                rates[str(g)] = 0.0
            else:
                rates[str(g)] = float(y_pred[mask].mean())
        gap = abs(list(rates.values())[1] - list(rates.values())[0])
        result[f"{metric}_gap"] = round(gap, 4)
        result[f"{metric}_rates"] = {str(k): round(v, 4) for k, v in rates.items()}

    logger.info("equalized_odds %s", result)
    return result


def compute_fairness_report(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_col: str = "fico_score",
    threshold: float = 700,
) -> dict:
    """Generate a fairness report splitting on a numeric feature threshold."""
    if sensitive_col not in X.columns:
        return {"error": f"{sensitive_col} not in features"}

    sensitive = (X[sensitive_col] >= threshold).astype(int).values
    dpd = demographic_parity_difference(y_pred, sensitive)
    eod = equalized_odds_difference(y_true, y_pred, sensitive)
    return {
        "sensitive_feature": sensitive_col,
        "threshold": threshold,
        "demographic_parity_difference": dpd,
        **eod,
    }
