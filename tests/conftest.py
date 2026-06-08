"""Shared pytest fixtures for Loan-Lens test suite."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

# ── In-memory SQLite for tests ────────────────────────────────────────────────
TEST_DB_URL = "sqlite:///./test_loan_lens.db"

engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def create_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_application() -> dict:
    return {
        "loan_amount": 10000.0,
        "annual_income": 60000.0,
        "installment": 320.0,
        "interest_rate": 12.5,
        "loan_term_months": 36,
        "fico_score": 700,
        "revolving_utilization": 0.25,
        "revolving_balance": 5000.0,
        "delinquencies_2y": 0,
        "credit_history_months": 84,
        "open_accounts": 5,
        "total_accounts": 12,
        "public_records": 0,
    }


@pytest.fixture
def synthetic_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
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
    y = pd.Series(rng.integers(0, 2, n))
    return X, y
