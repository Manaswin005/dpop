"""Test suite for Stage 2 implementation-stage data pilot.

Validates date ordering, zero test-set leakage, missing value semantics,
provenance traceability, and strict target separation.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.stage2.cohort import select_pilot_cohort
from src.stage2.parser import parse_dates_for_delay, parse_isr_document


def _parse_flex_date(val: str | None) -> pd.Timestamp | None:
    if val is None or pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str.isdigit():
        # Excel date serial
        return pd.to_datetime(int(val_str), unit="D", origin="1899-12-30")
    return pd.to_datetime(val_str, errors="coerce")


def test_pilot_cohort_train_only():
    """Verify pilot cohort is drawn strictly from train.csv with zero test overlap."""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    cohort = select_pilot_cohort()

    assert len(cohort) == 30
    assert set(cohort["Project ID"]).issubset(set(train["Project ID"]))
    assert len(set(cohort["Project ID"]).intersection(set(test["Project ID"]))) == 0


def test_pilot_cohort_reproducibility():
    """Verify cohort selection reproduces identically with seed 42."""
    c1 = select_pilot_cohort(random_seed=42)
    c2 = select_pilot_cohort(random_seed=42)
    assert c1["Project ID"].tolist() == c2["Project ID"].tolist()
    assert c1["outcome_class"].tolist() == c2["outcome_class"].tolist()


def test_schedule_delay_missing_returns_nan():
    """Verify schedule delay returns NaN (never 0.0) when closing dates are missing."""
    text_missing = "Key Dates\nBoard Approval: 10-Jan-2010\nNo closing dates listed."
    delay = parse_dates_for_delay(text_missing)
    assert np.isnan(delay)


def test_schedule_delay_identical_returns_zero():
    """Verify schedule delay returns 0.0 when original and revised dates are identical."""
    text_equal = "Original Closing Date:31-Dec-2015 Revised Closing Date:31-Dec-2015"
    delay = parse_dates_for_delay(text_equal)
    assert delay == 0.0


def test_schedule_delay_extended_returns_months():
    """Verify schedule delay computes correct month extension when dates are revised."""
    text_extended = "Original Closing Date:31-Dec-2015 Revised Closing Date:30-Jun-2017"
    delay = parse_dates_for_delay(text_extended)
    assert delay == 18.0


def test_snapshot_temporal_validity_and_provenance():
    """Verify valid snapshots precede IEG evaluation and succeed approval."""
    snap_path = Path("data/stage2/pilot_snapshots.csv")
    prov_path = Path("data/stage2/pilot_provenance.csv")
    cohort_path = Path("data/stage2/pilot_cohort.csv")

    if not snap_path.exists() or not prov_path.exists():
        pytest.skip("Pilot datasets not yet generated.")

    snaps = pd.read_csv(snap_path)
    prov = pd.read_csv(prov_path)
    cohort = pd.read_csv(cohort_path)

    assert len(snaps) > 0, "No valid snapshots found in pilot dataset."

    cohort_dates = cohort.set_index("Project ID")

    for _, row in snaps.iterrows():
        pid = row["project_id"]
        snap_d_str = row["snapshot_date"]
        assert pd.notna(snap_d_str), f"Snapshot date is null for {pid}"

        snap_dt = pd.to_datetime(str(snap_d_str)[:10])

        app_d_str = cohort_dates.loc[pid, "Approval Date"]
        eval_d_str = cohort_dates.loc[pid, "IEG_EvalDate"]

        app_dt = _parse_flex_date(app_d_str)
        eval_dt = _parse_flex_date(eval_d_str)

        if app_dt is not None:
            assert snap_dt >= app_dt, f"Snapshot {snap_dt} predates approval {app_dt} for {pid}"

        if eval_dt is not None:
            assert snap_dt <= eval_dt, f"Snapshot {snap_dt} succeeds evaluation {eval_dt} for {pid}"

    # Verify provenance correspondence
    assert set(snaps["project_id"]).issubset(set(prov["project_id"]))
    valid_prov = prov[prov["status"] == "VALID_INTERIM_SNAPSHOT"]
    assert len(snaps) == len(valid_prov)
