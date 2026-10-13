"""Phase 8C — Policy decision support (architecture Stage 13).

Portfolio-level view, separate from the per-project recommendations:
  - Low / Medium / High risk mix by sector, region and lending instrument
  - Rule-based policy considerations: supervision allocation, risk mitigation,
    planning cautions. Wording describes associations, never causes.
  - Optional: score a CSV of proposed projects with the existing predictor.

Data source for the historical portfolio
  The EBM's saved OUT-OF-FOLD predictions on the training set
  (artifacts/cv/EBM_oof.npy). Each project was scored by a model trained on the
  other four folds, i.e. a model that had not seen it. verify_oof() checks the
  file before use. The test set is not used. Risk levels come from the existing
  assign_risk() rules, unchanged.

Run:  python -m src.decision_support.policy   -> reports/policy_portfolio.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.decision_support.risk import RISK_LEVELS, assign_risk
from src.utils.config import get_logger, load_config, resolve

log = get_logger("policy")

DIMENSIONS = {"Sector": "Sector Board", "Region": "Region", "Lending instrument": "Lending Instrument"}
# Provisional, project-defined rule settings (not validated decision criteria)
MIN_PROJECTS = 30        # smaller segments are reported but get no policy consideration
MARGIN = 0.10            # a segment stands out if its share differs from the portfolio by >= 10 points


class PortfolioError(ValueError):
    """Raised when the out-of-fold file or an uploaded portfolio cannot be used."""


# ----------------------------------------------------------------------------
# Historical portfolio from verified out-of-fold predictions
# ----------------------------------------------------------------------------
def verify_oof(model: str = "EBM") -> np.ndarray:
    """Load the saved out-of-fold scores and check they belong to the training run."""
    cfg = load_config()
    cv_dir = resolve(cfg["paths"]["artifacts_dir"]) / "cv"
    oof_path, rec_path = cv_dir / f"{model}_oof.npy", cv_dir / f"{model}.json"
    if not oof_path.exists() or not rec_path.exists():
        raise PortfolioError(f"Out-of-fold predictions for {model} not found in {cv_dir}. "
                             "Run Phase 3 training (`python -m src.models.train train`).")
    oof, rec = np.load(oof_path), json.loads(rec_path.read_text())
    train = pd.read_csv(resolve(cfg["paths"]["processed_dir"]) / "train.csv")
    problems = []
    if oof.shape != (len(train), len(cfg["target"]["class_order"])):
        problems.append(f"shape {oof.shape} does not match {len(train)} training projects")
    elif not np.allclose(oof.sum(axis=1), 1, atol=1e-6):
        problems.append("rows do not sum to 1 (some projects were never scored)")
    else:
        # The per-fold scores recomputed from the file must equal those recorded during training
        from src.models.train import folds
        order = cfg["target"]["class_order"]
        y = train[cfg["columns"]["target"]].map(order.index).to_numpy()
        recomputed = [round(f1_score(y[va], oof[va].argmax(1), average="macro"), 4) for _, va in folds(y)]
        if recomputed != rec.get("cv_macro_f1_folds"):
            problems.append(f"fold scores {recomputed} differ from the training record {rec.get('cv_macro_f1_folds')}")
    if problems:
        raise PortfolioError(f"{oof_path.name} failed verification: " + "; ".join(problems))
    return oof


def historical_portfolio() -> pd.DataFrame:
    """Training projects with their out-of-fold risk level and actual outcome."""
    cfg = load_config()
    order = cfg["target"]["class_order"]
    oof = verify_oof("EBM")
    train = pd.read_csv(resolve(cfg["paths"]["processed_dir"]) / "train.csv")
    out = train[["Project ID", *DIMENSIONS.values()]].copy()
    out["risk_level"] = [assign_risk(p, order)["risk_level"] for p in oof]
    out["actual_outcome"] = train[cfg["columns"]["target"]]
    return out


# ----------------------------------------------------------------------------
# Risk profiles and policy rules
# ----------------------------------------------------------------------------
def risk_profile(portfolio: pd.DataFrame, by: str) -> pd.DataFrame:
    """Projects and Low/Medium/High shares per segment (plus actual Unsuccessful share if known)."""
    tab = pd.crosstab(portfolio[by], portfolio["risk_level"]).reindex(columns=RISK_LEVELS, fill_value=0)
    prof = tab.div(tab.sum(axis=1), axis=0)
    prof.insert(0, "Projects", tab.sum(axis=1))
    if "actual_outcome" in portfolio:
        prof["Actually Unsuccessful"] = portfolio.groupby(by)["actual_outcome"].apply(lambda s: (s == "Unsuccessful").mean())
    return prof.sort_values(["High", "Projects"], ascending=False)


def overall_mix(portfolio: pd.DataFrame) -> pd.Series:
    return portfolio["risk_level"].value_counts(normalize=True).reindex(RISK_LEVELS, fill_value=0.0)


def policy_considerations(profile: pd.DataFrame, overall: pd.Series, dimension: str) -> dict:
    """Rule-based considerations for one dimension (e.g. sector). Associations only."""
    big = profile[profile["Projects"] >= MIN_PROJECTS]
    small = profile.index[profile["Projects"] < MIN_PROJECTS].tolist()
    high = big[big["High"] >= overall["High"] + MARGIN]
    elevated = big[(big["High"] + big["Medium"]) >= (overall["High"] + overall["Medium"]) + MARGIN]
    elevated = elevated.drop(high.index, errors="ignore")
    low = big[big["Low"] >= overall["Low"] + MARGIN]
    mixed = high.index.intersection(low.index)          # many High AND many Low projects
    high, low = high.drop(mixed), low.drop(mixed)

    supervision = [f"{seg}: {r['High']:.0%} of projects are High risk (portfolio average {overall['High']:.0%}). "
                   f"Consider prioritising supervision and monitoring capacity for this {dimension.lower()}."
                   for seg, r in high.iterrows()]
    supervision += [f"{seg}: {r['Low']:.0%} of projects are Low risk (portfolio average {overall['Low']:.0%}). "
                    "Standard supervision may be sufficient, freeing capacity for higher-risk segments."
                    for seg, r in low.iterrows()]
    supervision += [f"{seg}: mixed picture, with {profile.loc[seg, 'High']:.0%} High risk and {profile.loc[seg, 'Low']:.0%} "
                    "Low risk projects. Consider allocating supervision case by case rather than to the whole segment."
                    for seg in mixed]
    # Each segment is described by the rule that flagged it
    mitigation = [f"{seg}: High risk for {r['High']:.0%} of projects (portfolio average {overall['High']:.0%}). "
                  "Consider readiness reviews at approval, early-warning indicators and planned mid-term reviews."
                  for seg, r in high.iterrows()]
    mitigation += [f"{seg}: High or Medium risk for {r['High'] + r['Medium']:.0%} of projects "
                   f"(portfolio average {overall['High'] + overall['Medium']:.0%}). Consider early readiness checks "
                   "and clear results-monitoring arrangements." for seg, r in elevated.iterrows()]
    cautions = ["These patterns are associations in historical model scores. They do not show that a "
                f"{dimension.lower()} causes better or worse outcomes, and they should not alone decide budgets."]
    if small:
        cautions.append(f"{len(small)} segment(s) have fewer than {MIN_PROJECTS} projects ({', '.join(map(str, small[:5]))}"
                        f"{', ...' if len(small) > 5 else ''}); too few to support a policy conclusion.")
    return {"supervision_allocation": supervision, "risk_mitigation": mitigation, "planning_cautions": cautions}


GENERAL_CAUTIONS = [
    "Risk levels use the project's provisional, documented risk rules; model scores are not real-world probabilities.",
    "Approval year strongly influences the model and partly reflects a change in IEG rating practice before FY1990.",
    "Budget and staffing decisions need expert review, country context and information the model does not see.",
]


# ----------------------------------------------------------------------------
# Optional: proposed-project portfolio from a CSV upload
# ----------------------------------------------------------------------------
def required_columns(predictor) -> list[str]:
    return [*predictor.cats, "Approval FY"]


def score_portfolio(df: pd.DataFrame, predictor) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each row with the existing predictor (no SHAP, for speed).

    Returns (scored rows, rejected rows with reasons). Invalid rows are never scored.
    """
    missing_cols = [c for c in required_columns(predictor) if c not in df.columns]
    if missing_cols:
        raise PortfolioError(f"The CSV is missing required column(s): {', '.join(missing_cols)}. "
                             f"Expected: {', '.join(required_columns(predictor))}, and optionally Lending Project Cost.")
    if df.empty:
        raise PortfolioError("The CSV has no rows.")
    from src.prediction.predictor import InputError

    scored, rejected = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        inputs = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            r = predictor.predict(inputs, with_explanation=False)
            scored.append({"row": i + 1, **{k: r["inputs"][k] for k in predictor.cats},
                           "Approval FY": r["inputs"]["Approval FY"],
                           "Region": inputs.get("Region"), "predicted_outcome": r["predicted_outcome"],
                           "risk_level": r["risk_level"], "risk_score": r["risk_score"],
                           "warnings": " | ".join(r["warnings"])})
        except InputError as e:
            rejected.append({"row": i + 1, "problems": "; ".join(e.messages)})
    return pd.DataFrame(scored), pd.DataFrame(rejected, columns=["row", "problems"])


def template_csv(predictor) -> str:
    """A one-row example file users can download and fill in."""
    from src.prediction.predictor import EXAMPLES
    return pd.DataFrame([EXAMPLES["Example B"]]).to_csv(index=False)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    pf = historical_portfolio()
    overall = overall_mix(pf)
    lines = ["# Stage 13 — Policy decision support (historical portfolio)", "",
             f"Source: EBM out-of-fold predictions for {len(pf):,} training projects, verified against the "
             "training record. Each project was scored by a model that had not seen it. The test set was not used.", "",
             f"Portfolio risk mix: Low {overall['Low']:.0%}, Medium {overall['Medium']:.0%}, High {overall['High']:.0%}.", "",
             f"Rule settings (provisional): segments need at least {MIN_PROJECTS} projects, and a share must differ "
             f"from the portfolio by at least {MARGIN * 100:.0f} percentage points to stand out.", ""]
    for label, col in DIMENSIONS.items():
        prof = risk_profile(pf, col)
        cons = policy_considerations(prof, overall, label)
        lines += [f"## By {label.lower()}", "",
                  "| Segment | Projects | Low | Medium | High | Actually Unsuccessful |", "|---|---|---|---|---|---|",
                  *[f"| {seg} | {int(r['Projects']):,} | {r['Low']:.0%} | {r['Medium']:.0%} | {r['High']:.0%} | "
                    f"{r['Actually Unsuccessful']:.0%} |" for seg, r in prof.iterrows()], ""]
        for title, key in [("Supervision allocation", "supervision_allocation"),
                           ("Risk mitigation", "risk_mitigation"), ("Planning cautions", "planning_cautions")]:
            lines += [f"**{title}**", *([f"- {t}" for t in cons[key]] or ["- No segment stands out."]), ""]
    lines += ["## General cautions", *[f"- {c}" for c in GENERAL_CAUTIONS]]
    (resolve(cfg["paths"]["reports_dir"]) / "policy_portfolio.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Policy report written to reports/policy_portfolio.md")


if __name__ == "__main__":
    main()
