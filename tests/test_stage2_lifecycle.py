"""Focused tests for Project Lifecycle Workflow:
- Project creation and unique project ID persistence
- Reopening an existing project
- Both stage paths: Before Implementation & During Implementation
- Entering the six monitoring indicators
- Prototype rule-based stance generation and dated record updates
- Baseline vs. actual comparison display
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from src.stage2 import lifecycle_store
from src.utils.config import resolve
from stage2_experiments.interim_risk_assessment import assess_interim_risk

APP = str(resolve("app.py"))


def test_lifecycle_store_crud(tmp_path: Path):
    """Test project creation, retrieval, and monitoring record updates in isolation."""
    store_file = tmp_path / "lifecycle_test.json"

    # 1. Create a project
    pid = lifecycle_store.create_or_update_project(
        project_id=None,
        title="Rural Clean Water Initiative",
        description="Providing potable water networks in drought-prone areas.",
        stage1_inputs={"Country Name": "Kenya", "Sector Board": "Water"},
        stage1_prediction={"predicted_outcome": "Partially Successful", "risk_level": "Medium", "risk_score": 0.42},
        created_date="2026-10-13",
        storage_path=store_file,
    )
    assert pid.startswith("PRJ-")

    # 2. Reopen and verify project details
    loaded = lifecycle_store.get_project(pid, storage_path=store_file)
    assert loaded is not None
    assert loaded["title"] == "Rural Clean Water Initiative"
    assert loaded["stage1_prediction"]["predicted_outcome"] == "Partially Successful"
    assert len(loaded["monitoring_records"]) == 0

    # 3. Add a dated monitoring record with the 6 indicators
    indicators = {
        "budget_utilization_pct": 35.0,
        "beneficiary_participation_pct": 75.0,
        "project_completion_pct": 28.0,
        "implementation_delay_months": 0.0,
        "resource_availability": "High / Adequate",
        "beneficiary_coverage_pct": 50.0,
    }
    assessment = assess_interim_risk(
        pdo_rating="Satisfactory",
        ip_rating="Satisfactory",
        disbursement_pct=35.0,
        delay_months=0.0,
        stage1_risk="Medium",
    )
    ok = lifecycle_store.add_monitoring_record(
        project_id=pid,
        record_date="2027-04-15",
        indicators=indicators,
        interim_assessment=assessment,
        notes="First annual supervision review.",
        storage_path=store_file,
    )
    assert ok is True

    # 4. Reopen and verify monitoring update
    updated = lifecycle_store.get_project(pid, storage_path=store_file)
    assert len(updated["monitoring_records"]) == 1
    rec = updated["monitoring_records"][0]
    assert rec["record_date"] == "2027-04-15"
    assert rec["indicators"]["budget_utilization_pct"] == 35.0
    assert rec["interim_assessment"]["interim_risk_stance"] == "Standard Supervision"


def test_app_lifecycle_before_implementation():
    """Verify Streamlit app renders lifecycle options and runs Stage 1 Before Implementation."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["page"] = "Predict a project"
    at.run()
    assert not at.exception

    # Check title and stage radio
    assert any("Project lifecycle" in t.value for t in at.title)
    assert at.radio(key="lifecycle_stage_radio").value == "Before Implementation"

    # Set project title
    at.text_input(key="in_proj_title").set_value("Test Irrigation Project")
    at.selectbox(key="in_country").set_value("Kenya")
    at.selectbox(key="in_sector").set_value("Water")
    at.selectbox(key="in_agreement").set_value("IDA")
    at.selectbox(key="in_instrument").set_value("SIL")

    # Submit Stage 1 form (button[0])
    at.button[0].click()
    at.run()
    assert not at.exception
    assert not at.error
    assert any("Project saved under ID" in s.value for s in at.success)
    assert at.session_state["last_result"]["predicted_outcome"] in ("Successful", "Partially Successful", "Unsuccessful")


def test_app_lifecycle_during_implementation_path():
    """Verify Streamlit app allows selecting During Implementation and submitting 6 indicators."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["page"] = "Predict a project"
    at.run()
    assert not at.exception

    # Switch radio to During Implementation
    at.radio(key="lifecycle_stage_radio").set_value("During Implementation")
    at.run()
    assert not at.exception

    # Verify Prototype Notice is present
    assert any("Prototype Notice" in w.value for w in at.warning)

    # Submit monitoring form (button[0] when in During Implementation mode)
    at.text_input(key="in_proj_title").set_value("Health Clinic Upgrades")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert not at.error
    assert any("Monitoring record saved" in s.value for s in at.success)
    assert "last_monitoring_result" in at.session_state
    mon_res = at.session_state["last_monitoring_result"]
    assert "budget_utilization_pct" in mon_res["indicators"]
    assert mon_res["assessment"]["interim_risk_stance"] in (
        "Standard Supervision", "Moderate Implementation Attention", "High Implementation Alert"
    )


def test_load_existing_project_by_exact_id():
    """Regression test: verify loading PRJ-E22E84B3 by exact ID restores title, inputs, and Stage 1 prediction."""
    # 1. Verify direct store retrieval
    prj = lifecycle_store.get_project("PRJ-E22E84B3")
    assert prj is not None, "PRJ-E22E84B3 not found in lifecycle_projects.json"
    assert prj["title"] == "Agriculture"
    assert prj["stage1_prediction"]["predicted_outcome"] == "Partially Successful"
    assert prj["stage1_inputs"]["Country Name"] == "Andean Countries"

    # 2. Verify Streamlit UI lookup restores project state
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["page"] = "Predict a project"
    at.run()
    assert not at.exception

    # Type exact ID in lookup input
    at.text_input(key="lifecycle_lookup_id").set_value("PRJ-E22E84B3")
    at.run()
    assert not at.exception
    assert not at.error

    # Verify restored UI fields and session state
    assert at.text_input(key="in_proj_title").value == "Agriculture"
    assert at.selectbox(key="in_country").value == "Andean Countries"
    assert at.selectbox(key="in_sector").value == "Agriculture and Rural Development"
    assert at.selectbox(key="in_agreement").value == "MONT"
    assert at.selectbox(key="in_instrument").value == "FIL"
    assert at.session_state.get("last_result") is not None
    assert at.session_state["last_result"]["predicted_outcome"] == "Partially Successful"
    assert at.session_state["last_result"]["risk_level"] == "Medium"

