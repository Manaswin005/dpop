"""Phase 8C tests: Stage 13 policy decision support."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.decision_support import policy
from src.prediction.predictor import EXAMPLES, Predictor
from src.utils.config import resolve

CAUSAL_WORDS = ("causes better", "will improve", "guarantee", "leads to", "because of the sector")


# ---- data source ----------------------------------------------------------
def test_oof_predictions_verified():
    oof = policy.verify_oof("EBM")
    train = pd.read_csv(resolve("data/processed/train.csv"))
    assert oof.shape == (len(train), 3) and np.allclose(oof.sum(axis=1), 1)


def test_corrupted_oof_is_rejected(tmp_path, monkeypatch):
    # A file that does not match the training record must not be used
    import shutil
    art = tmp_path / "artifacts" / "cv"
    art.mkdir(parents=True)
    shutil.copy(resolve("artifacts/cv/EBM.json"), art / "EBM.json")
    rng = np.random.default_rng(0)
    fake = rng.dirichlet([1, 1, 1], size=8600)
    np.save(art / "EBM_oof.npy", fake)
    real_resolve = policy.resolve
    monkeypatch.setattr(policy, "resolve",
                        lambda rel: tmp_path / rel if str(rel).startswith("artifacts") else real_resolve(rel))
    with pytest.raises(policy.PortfolioError, match="failed verification"):
        policy.verify_oof("EBM")


@pytest.fixture(scope="module")
def portfolio():
    return policy.historical_portfolio()


def test_historical_portfolio_uses_existing_risk_rules(portfolio):
    from src.decision_support.risk import assign_risk
    oof = policy.verify_oof("EBM")
    order = ["Successful", "Partially Successful", "Unsuccessful"]
    expected = [assign_risk(p, order)["risk_level"] for p in oof[:50]]
    assert portfolio["risk_level"].head(50).tolist() == expected


# ---- risk profiles --------------------------------------------------------
@pytest.mark.parametrize("column", list(policy.DIMENSIONS.values()))
def test_risk_profile_shares_and_counts(portfolio, column):
    prof = policy.risk_profile(portfolio, column)
    assert prof["Projects"].sum() == len(portfolio)
    assert np.allclose(prof[["Low", "Medium", "High"]].sum(axis=1), 1)


# ---- policy rules (synthetic profile so the expected result is known) -----
def test_policy_rules_on_known_profile():
    prof = pd.DataFrame({"Projects": [100, 100, 100, 10, 100],
                         "Low": [0.05, 0.60, 0.20, 0.0, 0.45],
                         "Medium": [0.45, 0.30, 0.55, 0.0, 0.05],
                         "High": [0.50, 0.10, 0.25, 1.0, 0.50]},
                        index=["Risky", "Safe", "Typical", "Tiny", "Mixed"])
    overall = pd.Series({"Low": 0.20, "Medium": 0.55, "High": 0.25})
    cons = policy.policy_considerations(prof, overall, "Sector")
    sup = " ".join(cons["supervision_allocation"])
    assert "Risky: 50% of projects are High risk" in sup
    assert "Safe: 60% of projects are Low risk" in sup
    assert "Mixed: mixed picture" in sup and "Mixed: 50% of projects are High risk" not in sup
    assert "Typical" not in sup                          # close to the portfolio average
    assert "Tiny" not in sup and "Tiny" in " ".join(cons["planning_cautions"])   # too few projects
    assert any(t.startswith("Risky: High risk") for t in cons["risk_mitigation"])
    # Advice never uses causal wording; the cautions state that patterns are associations
    advice = " ".join(cons["supervision_allocation"] + cons["risk_mitigation"]).lower()
    assert not any(w in advice for w in CAUSAL_WORDS)
    assert "association" in " ".join(cons["planning_cautions"]).lower()


# ---- optional CSV upload --------------------------------------------------
@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_score_portfolio_valid_and_invalid_rows(predictor):
    good = dict(EXAMPLES["Example B"])
    bad = dict(EXAMPLES["Example C"], **{"Approval FY": "not a year", "Sector Board": None})
    scored, rejected = policy.score_portfolio(pd.DataFrame([good, bad, EXAMPLES["Example L"]]), predictor)
    assert scored["row"].tolist() == [1, 3]
    assert rejected["row"].tolist() == [2] and "Approval FY" in rejected["problems"].iloc[0]
    # Same result as the single-project prediction service
    assert scored["risk_level"].iloc[0] == predictor.predict(good, with_explanation=False)["risk_level"]


def test_score_portfolio_missing_columns(predictor):
    with pytest.raises(policy.PortfolioError, match="missing required column"):
        policy.score_portfolio(pd.DataFrame({"Country Name": ["Kenya"]}), predictor)


def test_template_csv_is_valid_input(predictor):
    import io
    scored, rejected = policy.score_portfolio(pd.read_csv(io.StringIO(policy.template_csv(predictor))), predictor)
    assert len(scored) == 1 and rejected.empty


# ---- dashboard ------------------------------------------------------------
def test_policy_page_renders():
    at = AppTest.from_file(str(resolve("app.py")), default_timeout=120)
    at.session_state["page"] = "Policy decision support"
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Policy decision support"
    assert len(at.tabs) == 3
    assert any("out-of-fold" in i.value for i in at.info)
