"""Automated retraining pipeline (Airflow DAG + standalone runner)."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from airflow.decorators import dag, task

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


# ── Standalone retraining functions ──────────────────────────────────────────

def load_recent_predictions(db_url: str, limit: int = 5000) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT input_features, prediction FROM prediction_logs "
                "ORDER BY id DESC LIMIT :limit"
            ),
            {"limit": limit},
        )
        rows = result.fetchall()

    records = []
    for row in rows:
        feats = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        feats["target"] = row[1]
        records.append(feats)

    return pd.DataFrame(records)


def check_drift_trigger(db_url: str, threshold: int = 3) -> bool:
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM drift_logs WHERE drift_detected=1 "
                "AND created_at > datetime('now', '-7 days')"
            )
        )
        count = result.scalar()
    triggered = count >= threshold
    logger.info("Drift trigger check: %d drifted features (threshold=%d) -> %s", count, threshold, triggered)
    return triggered


def retrain(db_url: str | None = None, min_samples: int = 200) -> dict:
    from app.model import train_model

    db_url = db_url or "sqlite:///./loan_lens.db"

    try:
        df = load_recent_predictions(db_url, limit=5000)
    except Exception as exc:
        logger.warning("Could not load predictions for retraining: %s — using synthetic data", exc)
        df = _generate_synthetic(2000)

    if len(df) < min_samples:
        logger.warning("Insufficient data for retraining: %d < %d", len(df), min_samples)
        df = pd.concat([df, _generate_synthetic(min_samples - len(df))], ignore_index=True)

    target_col = "target" if "target" in df.columns else "prediction"
    y = df[target_col].astype(int)
    raw_cols = [
        "loan_amount", "annual_income", "installment", "interest_rate",
        "loan_term_months", "fico_score", "revolving_utilization", "revolving_balance",
        "delinquencies_2y", "credit_history_months", "open_accounts", "total_accounts",
        "public_records",
    ]
    available = [c for c in raw_cols if c in df.columns]
    X = df[available].fillna(0)

    _, metrics = train_model(X, y)
    logger.info("Retraining complete: AUC=%.4f", metrics.get("auc_mean", 0))
    return metrics


def _generate_synthetic(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
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
        "target": rng.integers(0, 2, n),
    })


# ── Airflow DAG (no-op if Airflow not installed) ──────────────────────────────
if AIRFLOW_AVAILABLE:
    @dag(
        dag_id="loan_lens_retrain",
        schedule="@weekly",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
        tags=["loan-lens", "ml", "retraining"],
    )
    def loan_lens_retrain_dag():
        @task
        def check_drift() -> bool:
            import os
            return check_drift_trigger(os.getenv("DATABASE_URL", "sqlite:///./loan_lens.db"))

        @task
        def run_retrain(should_retrain: bool) -> dict:
            import os
            if not should_retrain:
                logger.info("No drift detected — skipping retraining")
                return {}
            return retrain(os.getenv("DATABASE_URL"))

        @task
        def log_result(metrics: dict) -> None:
            if metrics:
                logger.info("DAG retraining result: %s", metrics)

        log_result(run_retrain(check_drift()))

    loan_lens_retrain_dag()
