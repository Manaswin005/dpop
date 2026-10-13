"""Focused unit tests for Stage 2 exploratory experiment artifacts."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_exploratory_table_structure():
    """Verify pilot comparison table has 14 records, no leakage, and valid probabilities."""
    csv_path = Path("stage2_experiments/pilot_comparison_table.csv")
    assert csv_path.exists(), "pilot_comparison_table.csv does not exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 14

    # Check that Stage 1 probabilities are well-formed
    probs = df[["stage1_p_successful", "stage1_p_partially", "stage1_p_unsuccessful"]].sum(axis=1)
    assert np.allclose(probs, 1.0, atol=0.01)

    # Check all three outcome classes are represented
    counts = df["actual_outcome"].value_counts()
    assert "Unsuccessful" in counts and counts["Unsuccessful"] == 2
    assert "Partially Successful" in counts and counts["Partially Successful"] == 9
    assert "Successful" in counts and counts["Successful"] == 3


def test_exploratory_predictor_separation():
    """Verify predictor features do not contain outcome labels or raw metadata."""
    csv_path = Path("stage2_experiments/pilot_comparison_table.csv")
    df = pd.read_csv(csv_path)

    predictor_cols = [
        "pdo_rating_current",
        "ip_rating_current",
        "disbursement_percentage",
        "schedule_delay_months",
    ]
    for col in predictor_cols:
        assert col in df.columns
        assert not df[col].astype(str).str.contains("Partially Successful|Successful|Unsuccessful").any()
