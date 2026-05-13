"""Parametrized and edge-case API tests for Loan-Lens."""

import pytest


@pytest.mark.parametrize("fico_score", [300, 580, 700, 800, 850])
def test_predict_fico_range(client, sample_application, fico_score):
    app_data = dict(sample_application)
    app_data["fico_score"] = fico_score
    resp = client.post("/api/v1/predict", json=app_data)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["probability"] <= 1.0


@pytest.mark.parametrize("interest_rate", [1.0, 10.0, 20.0, 39.9])
def test_predict_interest_rate_range(client, sample_application, interest_rate):
    app_data = dict(sample_application)
    app_data["interest_rate"] = interest_rate
    resp = client.post("/api/v1/predict", json=app_data)
    assert resp.status_code == 200


@pytest.mark.parametrize("loan_term", [12, 24, 36, 60, 84])
def test_predict_loan_term_range(client, sample_application, loan_term):
    app_data = dict(sample_application)
    app_data["loan_term_months"] = loan_term
    resp = client.post("/api/v1/predict", json=app_data)
    assert resp.status_code == 200


@pytest.mark.parametrize("revolving_util", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_predict_utilization_range(client, sample_application, revolving_util):
    app_data = dict(sample_application)
    app_data["revolving_utilization"] = revolving_util
    resp = client.post("/api/v1/predict", json=app_data)
    assert resp.status_code == 200


def test_predict_minimum_valid_application(client):
    minimal = {
        "loan_amount": 1.0,
        "annual_income": 1.0,
        "installment": 1.0,
        "interest_rate": 1.0,
        "loan_term_months": 12,
        "fico_score": 300,
        "revolving_utilization": 0.0,
        "revolving_balance": 0.0,
        "delinquencies_2y": 0,
        "credit_history_months": 0,
        "open_accounts": 0,
        "total_accounts": 0,
        "public_records": 0,
    }
    resp = client.post("/api/v1/predict", json=minimal)
    assert resp.status_code == 200


def test_explain_endpoint(client, sample_application):
    resp = client.post("/api/v1/explain", json=sample_application)
    assert resp.status_code == 200
    data = resp.json()
    assert "top_features" in data
    assert "method" in data


@pytest.mark.parametrize("missing_field", ["loan_amount", "fico_score", "annual_income"])
def test_predict_missing_required_fields(client, sample_application, missing_field):
    app_data = dict(sample_application)
    del app_data[missing_field]
    resp = client.post("/api/v1/predict", json=app_data)
    assert resp.status_code == 422
