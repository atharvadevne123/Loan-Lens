"""Model drift detection and prediction logging."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path("reference_data.json")


def log_prediction(
    db: Session,
    correlation_id: str,
    input_features: dict,
    probability: float,
    prediction: int,
    model_version: str = "1.0.0",
) -> PredictionLog:
    record = PredictionLog(
        correlation_id=correlation_id,
        input_features=input_features,
        probability=probability,
        prediction=prediction,
        model_version=model_version,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug("Logged prediction id=%d corr=%s prob=%.4f", record.id, correlation_id, probability)
    return record


def compute_drift(reference: list[float], current: list[float]) -> dict:
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def compute_feature_drift(
    db: Session,
    window: int = 100,
) -> list[dict]:
    if not REFERENCE_PATH.exists():
        return []

    ref_df = pd.DataFrame(json.loads(REFERENCE_PATH.read_text()))

    recent = (
        db.query(PredictionLog)
        .order_by(PredictionLog.id.desc())
        .limit(window)
        .all()
    )
    if len(recent) < 10:
        return []

    current_records = [r.input_features for r in recent]
    current_df = pd.DataFrame(current_records)

    results = []
    shared_cols = [c for c in ref_df.columns if c in current_df.columns]
    for col in shared_cols:
        ref_vals = ref_df[col].dropna().tolist()
        cur_vals = current_df[col].dropna().tolist()
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue
        drift = compute_drift(ref_vals, cur_vals)
        drift["feature"] = col
        results.append(drift)

        if drift["drift_detected"]:
            record = DriftLog(
                feature=col,
                ks_statistic=drift["ks_statistic"],
                p_value=drift["p_value"],
                drift_detected=1,
            )
            db.add(record)
    db.commit()

    return results


def get_model_metrics() -> dict:
    metrics_path = Path("metrics.json")
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def compute_prediction_stats(db: Session, window: int = 1000) -> dict:
    recent = (
        db.query(PredictionLog)
        .order_by(PredictionLog.id.desc())
        .limit(window)
        .all()
    )
    if not recent:
        return {"total_predictions": 0}

    probs = [r.probability for r in recent]
    preds = [r.prediction for r in recent]
    return {
        "total_predictions": len(recent),
        "default_rate": round(sum(preds) / len(preds), 4),
        "avg_probability": round(float(np.mean(probs)), 4),
        "p50_probability": round(float(np.percentile(probs, 50)), 4),
        "p95_probability": round(float(np.percentile(probs, 95)), 4),
    }
