"""Tests for explainability module."""

import pytest

from app.explainability import get_shap_explanation, _finite_diff_explanation


def test_explain_returns_top_features(sample_application):
    result = get_shap_explanation(sample_application)
    assert "top_features" in result
    assert "method" in result
    assert len(result["top_features"]) <= 5


def test_explain_method_valid(sample_application):
    result = get_shap_explanation(sample_application)
    assert result["method"] in ("shap", "finite_diff")


def test_finite_diff_explanation_structure(sample_application):
    result = _finite_diff_explanation(sample_application)
    assert "method" in result
    assert result["method"] == "finite_diff"
    assert isinstance(result["top_features"], list)


def test_finite_diff_feature_names(sample_application):
    result = _finite_diff_explanation(sample_application)
    for item in result["top_features"]:
        assert "feature" in item
        assert "contribution" in item
        assert isinstance(item["contribution"], float)


@pytest.mark.parametrize("fico_score", [580, 700, 800])
def test_explain_various_fico_scores(sample_application, fico_score):
    app = dict(sample_application)
    app["fico_score"] = fico_score
    result = get_shap_explanation(app)
    assert "top_features" in result
    assert len(result["top_features"]) > 0
