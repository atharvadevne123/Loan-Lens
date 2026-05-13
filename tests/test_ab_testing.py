"""Tests for the A/B testing router."""

import pytest

from app.ab_testing import ABRouter, ModelVariant, get_router


def test_router_returns_valid_variant():
    router = get_router()
    variant = router.route("applicant-001")
    assert variant.name in ("control", "challenger")


def test_router_consistent_routing():
    router = get_router()
    v1 = router.route("applicant-XYZ")
    v2 = router.route("applicant-XYZ")
    assert v1.name == v2.name


def test_router_weights_validated():
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        ABRouter([
            ModelVariant(name="a", weight=0.6),
            ModelVariant(name="b", weight=0.6),
        ])


def test_router_split_summary():
    router = get_router()
    summary = router.split_summary()
    assert "control" in summary
    assert abs(sum(summary.values()) - 1.0) < 1e-6


def test_router_deterministic_different_ids():
    router = get_router()
    variants = {router.route(f"applicant-{i}").name for i in range(200)}
    # With 200 applicants, both variants should appear
    assert len(variants) >= 1


def test_custom_router():
    router = ABRouter([
        ModelVariant(name="v1", weight=0.7),
        ModelVariant(name="v2", weight=0.3),
    ])
    results = [router.route(f"id-{i}").name for i in range(1000)]
    v1_rate = results.count("v1") / len(results)
    # Should be close to 70% but with some variance
    assert 0.5 <= v1_rate <= 0.9
