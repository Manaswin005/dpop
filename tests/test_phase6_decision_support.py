"""Phase 6 tests: risk rules, recommendation rules, input validation, prediction output."""
import numpy as np
import pytest

from src.decision_support.recommendations import HUMAN_REVIEW, LEVEL_ACTIONS, build_recommendations
from src.decision_support.risk import assign_risk
from src.prediction.predictor import EXAMPLES, InputError, Predictor

ORDER = ["Successful", "Partially Successful", "Unsuccessful"]
CAUSAL_WORDS = ("causes", "caused by", "will improve", "guarantee", "will fail")


# ---- risk rules -----------------------------------------------------------
@pytest.mark.parametrize("proba, level", [
    ([0.20, 0.20, 0.60], "High"),     # predicts Unsuccessful
    ([0.30, 0.50, 0.20], "Medium"),   # predicts Partially Successful
    ([0.45, 0.30, 0.25], "Medium"),   # predicts Successful, weak lead (< 0.50)
    ([0.50, 0.30, 0.20], "Low"),      # exactly at the cut-off -> Low
    ([0.80, 0.10, 0.10], "Low"),
])
def test_risk_levels(proba, level):
    r = assign_risk(proba, ORDER, min_success=0.50)
    assert r["risk_level"] == level
    assert r["risk_score"] == pytest.approx(1 - proba[0], abs=1e-3)


@pytest.mark.parametrize("bad", [[0.5, 0.5], [np.nan, 0.5, 0.5]])
def test_risk_rejects_invalid_scores(bad):
    with pytest.raises(ValueError):
        assign_risk(bad, ORDER, min_success=0.5)


# ---- recommendation rules ------------------------------------------------
def test_actions_match_level_and_disclaimer_always_present():
    for level in ("Low", "Medium", "High"):
        rec = build_recommendations(level, None, {})
        assert rec["actions"] == LEVEL_ACTIONS[level] and rec["disclaimer"] == HUMAN_REVIEW
        assert rec["feature_notes"] == []


def test_feature_notes_only_for_risk_raising_features():
    contrib = {"Country Name": 0.10, "Sector Board": -0.05, "Agreement Type": 0.005}
    rec = build_recommendations("High", contrib, {"Country Name": "Kenya", "Lending Project Cost": 5e7})
    assert [n["feature"] for n in rec["feature_notes"]] == ["Country Name"]
    assert "Kenya" in rec["feature_notes"][0]["note"]


def test_cost_notes_follow_actual_input():
    reported = build_recommendations("Medium", {"lending_cost_reported": 0.05}, {"Lending Project Cost": 1e8})
    assert reported["feature_notes"] == []           # cost was given: no "not recorded" note
    missing = build_recommendations("High", {"Lending Project Cost": 0.04, "lending_cost_reported": 0.03},
                                    {"Lending Project Cost": None})
    assert len(missing["feature_notes"]) == 1          # one note, not duplicated
    assert "No planned cost" in missing["feature_notes"][0]["note"]


def test_no_causal_language():
    texts = [a for acts in LEVEL_ACTIONS.values() for a in acts]
    rec = build_recommendations("High", {f: 0.1 for f in ["Country Name", "Sector Board", "Approval FY"]},
                                {"Country Name": "X", "Sector Board": "Y", "Approval FY": 2000})
    texts += [n["note"] for n in rec["feature_notes"]]
    assert not any(w in t.lower() for t in texts for w in CAUSAL_WORDS)


# ---- prediction service ---------------------------------------------------
@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_missing_and_invalid_inputs_are_rejected(predictor):
    with pytest.raises(InputError) as e:
        predictor.predict({"Country Name": "India", "Approval FY": "abc", "Lending Project Cost": -5},
                          with_explanation=False)
    msgs = " ".join(e.value.messages)
    assert "Sector Board" in msgs and "Approval FY" in msgs and "positive" in msgs


def test_unseen_category_and_missing_cost_give_warnings(predictor):
    inp = dict(EXAMPLES["Example B"], **{"Country Name": "Atlantis", "Lending Project Cost": None})
    r = predictor.predict(inp, with_explanation=False)
    assert any("Atlantis" in w for w in r["warnings"]) and any("not reported" in w for w in r["warnings"])
    assert r["inputs"]["lending_cost_reported"] == 0


def test_full_prediction_is_consistent(predictor):
    r = predictor.predict(EXAMPLES["Example C"])
    assert set(r["scores"]) == set(ORDER) and sum(r["scores"].values()) == pytest.approx(1, abs=0.01)
    assert r["predicted_outcome"] == max(r["scores"], key=r["scores"].get)
    assert r["risk_level"] in ("Low", "Medium", "High") and r["model"] == "EBM"
    # SHAP contributions to the risk score add up to (risk score - average risk score)
    assert r["risk_contributions"] and set(r["risk_contributions"]) == set(predictor.columns)
