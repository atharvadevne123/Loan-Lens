"""API endpoint tests for Loan-Lens."""

import pytest


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_predict_success(client, sample_application):
    resp = client.post("/api/v1/predict", json=sample_application)
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data
    assert "prediction" in data
    assert "risk_level" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert data["risk_level"] in ("low", "medium", "high")


def test_predict_high_risk(client):
    app_data = {
        "loan_amount": 40000.0,
        "annual_income": 20000.0,
        "installment": 1500.0,
        "interest_rate": 30.0,
        "loan_term_months": 60,
        "fico_score": 580,
        "revolving_utilization": 0.99,
        "revolving_balance": 30000.0,
        "delinquencies_2y": 5,
        "credit_history_months": 12,
        "open_accounts": 15,
        "total_accounts": 15,
        "public_records": 3,
    }
    resp = client.post("/api/v1/predict", json=app_data)
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data


def test_predict_invalid_fico(client, sample_application):
    bad = dict(sample_application)
    bad["fico_score"] = 200  # below minimum 300
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_loan_amount(client, sample_application):
    bad = dict(sample_application)
    bad["loan_amount"] = -500  # must be > 0
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422


def test_predict_total_less_than_open(client, sample_application):
    bad = dict(sample_application)
    bad["open_accounts"] = 10
    bad["total_accounts"] = 5  # violates validator
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_endpoint(client, sample_application):
    # Make a prediction first so stats are non-empty
    client.post("/api/v1/predict", json=sample_application)
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert data["predictions"]["total_predictions"] >= 1


def test_drift_endpoint(client):
    resp = client.get("/api/v1/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "features_checked" in data
    assert "features_drifted" in data


def test_correlation_id_in_response(client, sample_application):
    resp = client.post("/api/v1/predict", json=sample_application)
    assert "x-correlation-id" in resp.headers


@pytest.mark.parametrize(
    "loan_amount,expected_status",
    [
        (500000.0, 200),
        (0.0, 422),
        (500001.0, 422),
    ],
)
def test_predict_loan_amount_boundaries(client, sample_application, loan_amount, expected_status):
    app_data = dict(sample_application)
    app_data["loan_amount"] = loan_amount
    resp = client.post("/api/v1/predict", json=app_data)
    assert resp.status_code == expected_status
