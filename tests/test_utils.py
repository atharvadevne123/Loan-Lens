"""Tests for utility functions."""

import pytest

from app.utils import clamp, format_probability, risk_tier, stable_hash


def test_stable_hash_deterministic():
    d = {"loan_amount": 1000, "fico_score": 700}
    assert stable_hash(d) == stable_hash(d)


def test_stable_hash_different():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_clamp_within():
    assert clamp(0.5, 0, 1) == 0.5


def test_clamp_below():
    assert clamp(-1, 0, 1) == 0


def test_clamp_above():
    assert clamp(2, 0, 1) == 1


@pytest.mark.parametrize("prob,expected", [(0.1, "low"), (0.5, "medium"), (0.8, "high")])
def test_risk_tier(prob, expected):
    assert risk_tier(prob) == expected


def test_format_probability_clamps():
    assert format_probability(1.5) == 1.0
    assert format_probability(-0.1) == 0.0
