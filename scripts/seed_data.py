"""Seed the database with synthetic loan predictions for testing and demos."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from app.database import SessionLocal, init_db
from app.model import _seed_model, predict
from app.monitoring import log_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def seed(n: int = 200, seed_value: int = 42) -> None:
    """Insert n synthetic predictions into the database."""
    init_db()
    _seed_model()

    rng = np.random.default_rng(seed_value)
    db = SessionLocal()

    try:
        for i in range(n):
            features = {
                "loan_amount": float(rng.uniform(1000, 40000)),
                "annual_income": float(rng.uniform(20000, 150000)),
                "installment": float(rng.uniform(50, 1500)),
                "interest_rate": float(rng.uniform(5, 30)),
                "loan_term_months": int(rng.choice([36, 60])),
                "fico_score": int(rng.integers(580, 850)),
                "revolving_utilization": float(rng.uniform(0, 1)),
                "revolving_balance": float(rng.uniform(0, 50000)),
                "delinquencies_2y": int(rng.integers(0, 5)),
                "credit_history_months": int(rng.integers(12, 360)),
                "open_accounts": int(rng.integers(2, 20)),
                "total_accounts": int(rng.integers(5, 40)),
                "public_records": int(rng.integers(0, 3)),
            }
            result = predict(features)
            log_prediction(
                db=db,
                correlation_id=f"seed-{i:04d}",
                input_features=features,
                probability=result["probability"],
                prediction=result["prediction"],
            )
        logger.info("Seeded %d predictions into database", n)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Loan-Lens DB with synthetic predictions")
    parser.add_argument("-n", type=int, default=200, help="Number of predictions to insert")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    seed(n=args.n, seed_value=args.seed)
