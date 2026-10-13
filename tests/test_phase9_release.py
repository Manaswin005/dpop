"""Release-audit regression tests: dashboard figures come from saved results, not hard-coded text."""
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.app import data
from src.utils.config import resolve

APP = resolve("app.py")


def _comp(**overrides):
    df = pd.DataFrame({"model": ["EBM", "Ensemble (soft voting)", "GPC"],
                       "cv_macro_f1": [0.53, 0.52, 0.46], "test_macro_f1": [0.54, 0.50, 0.45],
                       "train_rows_used": [8600, 8600, 2000]})
    for k, v in overrides.items():
        df.loc[df["model"] == k, "test_macro_f1"] = v
    return df


def test_notes_only_mention_a_higher_test_score_when_true():
    meta = {"selected_model": "EBM"}
    assert not any("higher test macro-F1" in n for n in data.performance_notes(_comp(), meta))
    notes = data.performance_notes(_comp(**{"Ensemble (soft voting)": 0.60}), meta)
    assert any("0.600 vs 0.540" in n and "EBM leads (0.530 vs 0.520)" in n for n in notes)


def test_notes_report_subsample_and_missing_files():
    notes = data.performance_notes(_comp(), {"selected_model": "EBM", "missing_model_files": ["RuleFit"]})
    assert "GPC was trained on a 2,000-row subsample." in notes
    assert any("RuleFit was not evaluated" in n for n in notes)


def test_app_has_no_hard_coded_result_numbers():
    src = APP.read_text(encoding="utf-8")
    for stale in ("0.542", "0.540", "2,150 projects", "8,600 projects", "2,000-row"):
        assert stale not in src


def test_performance_page_caption_follows_saved_results(monkeypatch):
    comp = data.model_comparison()
    sel = data.model_metadata()["selected_model"]
    changed = comp.copy()
    changed.loc[changed["model"] == sel, "test_macro_f1"] = 0.123      # pretend results changed
    monkeypatch.setattr(data, "model_comparison", lambda: changed)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["page"] = "Model performance"
    at.run()
    assert not at.exception
    assert any(f"vs 0.123" in c.value for c in at.caption)
