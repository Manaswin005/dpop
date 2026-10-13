"""Focused unit tests for Phase 7 Stage 2 interim assessment demonstration."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from stage2_experiments.interim_risk_assessment import assess_interim_risk


def test_interim_assessment_demo_table_structure():
    """Verify demo table exists, has 14 rows, and valid columns."""
    demo_csv = Path("stage2_experiments/stage2_interim_assessment_demo.csv")
    assert demo_csv.exists(), "stage2_interim_assessment_demo.csv not found"

    df = pd.read_csv(demo_csv)
    assert len(df) == 14
    expected_cols = [
        "project_id",
        "snapshot_date",
        "pdo_rating",
        "ip_rating",
        "disbursement_pct",
        "schedule_delay_months",
        "stage1_baseline_risk",
        "interim_risk_stance",
        "interim_signals",
        "eventual_outcome_target",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify illustrative stances
    valid_stances = {"High Implementation Alert", "Moderate Implementation Attention", "Standard Supervision"}
    assert set(df["interim_risk_stance"]).issubset(valid_stances)


def test_assess_interim_risk_logic():
    """Test explicit rule evaluation with missing and present signals."""
    # Test distress rating
    res_distress = assess_interim_risk("Moderately Unsatisfactory", "Satisfactory", 10.0, 0.0, "Medium")
    assert res_distress["interim_risk_stance"] == "High Implementation Alert"
    assert "PDO rating downgraded" in res_distress["interim_signals_summary"]

    # Test clean interim
    res_clean = assess_interim_risk("Satisfactory", "Satisfactory", 30.0, 0.0, "Low")
    assert res_clean["interim_risk_stance"] == "Standard Supervision"

    # Test explicit null handling (no crash, clean stance)
    res_null = assess_interim_risk(None, None, None, None, "Medium")
    assert res_null["interim_risk_stance"] == "Standard Supervision"
    assert res_null["interim_signals_summary"] == "No adverse flags identified"
