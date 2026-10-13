"""Focused test for Stage 2 prototype dashboard page rendering."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest
from src.utils.config import resolve

APP = str(resolve("app.py"))


def test_stage2_prototype_page_renders_cleanly():
    """Verify Stage 2 prototype page renders without exceptions."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["page"] = "Implementation-stage risk (prototype)"
    at.run()
    assert not at.exception
    assert any("Implementation-stage risk assessment" in t.value for t in at.title)
    assert not at.error
