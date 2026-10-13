"""Phase 7 tests: every dashboard page renders, the form works, errors are friendly."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.app import data
from src.utils.config import resolve

APP = str(resolve("app.py"))
PAGES = ["Predict a project", "Explainability", "Historical insights", "Model performance"]


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["page"] = page
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    at = run_page(page)
    assert not at.exception, at.exception
    assert at.title[0].value  # every page has a title


def test_prediction_form_happy_path():
    at = run_page("Predict a project")
    at.selectbox(key="in_country").set_value("Kenya")
    at.selectbox(key="in_sector").set_value("Water")
    at.selectbox(key="in_agreement").set_value("IDA")
    at.selectbox(key="in_instrument").set_value("SIL")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert not at.error
    assert any(m.label == "Predicted outcome" for m in at.metric)
    assert at.session_state["last_result"]["risk_level"] in ("Low", "Medium", "High")


def test_prediction_form_shows_friendly_errors_when_fields_missing():
    at = run_page("Predict a project")
    at.button[0].click()          # submit with nothing selected
    at.run()
    assert not at.exception
    assert at.error and "Country Name" in at.error[0].value and "required" in at.error[0].value


def test_performance_page_reports_tabpfn_unavailable():
    at = run_page("Model performance")
    assert any("TabPFN" in w.value for w in at.warning)
    assert any("Selected model: **EBM**" in s.value for s in at.success)


def test_missing_model_is_handled(monkeypatch):
    # Simulate a missing/broken model: the app must show an error, not crash.
    from src.prediction import predictor as pred_module

    def broken_init(self):
        raise FileNotFoundError("Model metadata missing. Run Phases 3–4 first (see README).")

    monkeypatch.setattr(pred_module.Predictor, "__init__", broken_init)
    st.cache_resource.clear()   # drop the model cached by earlier tests
    at = run_page("Predict a project")
    st.cache_resource.clear()
    assert not at.exception
    assert at.error and "could not be loaded" in at.error[0].value


def test_outcome_mix_helper():
    df = data.training_data()
    mix = data.outcome_mix(df, "Sector Board")
    assert list(mix.columns) == ["Projects", *data.CLASS_ORDER]
    assert mix["Projects"].sum() == len(df)
    assert (mix[data.CLASS_ORDER].sum(axis=1) - 1).abs().max() < 0.01
