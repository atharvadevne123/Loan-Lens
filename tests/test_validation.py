"""Tests for the feature validation pipeline."""

import numpy as np
import pandas as pd
import pytest

from pipelines.feature_validation import validate_dataframe


@pytest.fixture
def valid_df(synthetic_dataframe):
    X, _ = synthetic_dataframe
    return X


def test_valid_dataframe_passes(valid_df):
    report = validate_dataframe(valid_df)
    assert report["valid"] is True
    assert report["issues"] == []


def test_missing_values_detected(valid_df):
    dirty = valid_df.copy()
    dirty.loc[0, "fico_score"] = np.nan
    report = validate_dataframe(dirty)
    assert not report["valid"]
    assert any("missing_values" in i for i in report["issues"])


def test_out_of_range_detected(valid_df):
    dirty = valid_df.copy()
    dirty.loc[0, "fico_score"] = 9999  # above max 850
    report = validate_dataframe(dirty)
    assert not report["valid"]
    assert any("out_of_range" in i and "fico_score" in i for i in report["issues"])


def test_missing_column_detected(valid_df):
    dirty = valid_df.drop(columns=["fico_score"])
    report = validate_dataframe(dirty)
    assert not report["valid"]
    assert any("missing_column" in i for i in report["issues"])


def test_report_contains_row_count(valid_df):
    report = validate_dataframe(valid_df)
    assert report["n_rows"] == len(valid_df)


def test_duplicate_rows_detected(valid_df):
    duped = pd.concat([valid_df, valid_df.iloc[:5]], ignore_index=True)
    report = validate_dataframe(duped)
    assert any("duplicate_rows" in i for i in report["issues"])
