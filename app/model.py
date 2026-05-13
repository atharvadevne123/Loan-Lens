"""ML model training, persistence, and prediction for Loan-Lens."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.features import CreditFeatureEngineer, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model.joblib")
METRICS_PATH = Path("metrics.json")
REFERENCE_PATH = Path("reference_data.json")

RAW_FEATURE_COLS = [
    "loan_amount", "annual_income", "installment", "interest_rate",
    "loan_term_months", "fico_score", "revolving_utilization", "revolving_balance",
    "delinquencies_2y", "credit_history_months", "open_accounts", "total_accounts",
    "public_records",
]


def build_ensemble() -> VotingClassifier:
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
    )


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    engineer = CreditFeatureEngineer()
    X_eng = engineer.fit_transform(X)
    X_feat = X_eng[FEATURE_COLUMNS]

    ensemble = build_ensemble()
    pipe = Pipeline([("scaler", StandardScaler()), ("model", ensemble)])

    cv_scores = cross_val_score(pipe, X_feat, y, cv=5, scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_feat, y)

    metrics = {
        "auc_mean": round(float(cv_scores.mean()), 4),
        "auc_std": round(float(cv_scores.std()), 4),
        "n_features": X_feat.shape[1],
        "n_train_samples": len(y),
        "default_rate": round(float(y.mean()), 4),
    }
    joblib.dump({"pipeline": pipe, "engineer": engineer}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # Save reference data for drift detection
    ref_sample = X_feat.sample(min(500, len(X_feat)), random_state=42)
    REFERENCE_PATH.write_text(ref_sample.to_json(orient="records"))

    logger.info("Model trained: AUC=%.4f±%.4f", metrics["auc_mean"], metrics["auc_std"])
    return pipe, metrics


def load_model() -> tuple[Pipeline, CreditFeatureEngineer]:
    if not MODEL_PATH.exists():
        _seed_model()
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["engineer"]


def predict(features: dict) -> dict:
    pipe, engineer = load_model()
    df = pd.DataFrame([features])
    X_eng = engineer.transform(df)
    X_feat = X_eng[FEATURE_COLUMNS]

    proba = float(pipe.predict_proba(X_feat)[0][1])
    label = int(proba >= 0.5)
    risk = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"

    return {"probability": round(proba, 4), "prediction": label, "risk_level": risk}


def _seed_model() -> None:
    """Generate synthetic training data to seed the model on first boot."""
    rng = np.random.default_rng(42)
    n = 2000
    data = pd.DataFrame({
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
    })
    y = pd.Series((
        (data["debt_to_income"] if "debt_to_income" in data.columns else data["loan_amount"] / data["annual_income"]) > 0.3
    ).astype(int) if False else rng.integers(0, 2, n))
    train_model(data, y)
