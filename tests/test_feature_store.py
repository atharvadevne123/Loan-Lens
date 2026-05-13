"""Tests for the LRU feature store."""

import time

import pytest

from app.feature_store import FeatureStore


@pytest.fixture
def store():
    s = FeatureStore(max_size=5, ttl=60)
    yield s
    s.clear()


def test_set_and_get(store):
    store.set("user1", {"fico_score": 700})
    result = store.get("user1")
    assert result == {"fico_score": 700}


def test_get_missing_returns_none(store):
    assert store.get("nonexistent") is None


def test_evicts_oldest_when_full(store):
    for i in range(6):
        store.set(f"user{i}", {"i": i})
    assert store.size <= 5


def test_ttl_expiry(store):
    fast_store = FeatureStore(max_size=10, ttl=0)
    fast_store.set("user1", {"data": 1})
    time.sleep(0.01)
    assert fast_store.get("user1") is None


def test_delete(store):
    store.set("user1", {"data": 1})
    store.delete("user1")
    assert store.get("user1") is None


def test_clear(store):
    store.set("u1", {})
    store.set("u2", {})
    store.clear()
    assert store.size == 0


def test_overwrite_existing(store):
    store.set("user1", {"v": 1})
    store.set("user1", {"v": 2})
    assert store.get("user1") == {"v": 2}
