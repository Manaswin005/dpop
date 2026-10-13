"""Phase 6 — Risk categories (architecture Stage 9).

Rules (provisional, project-defined; not validated decision thresholds):
  HIGH    the model predicts "Unsuccessful".
  MEDIUM  the model predicts "Partially Successful", OR it predicts "Successful"
          but its score for Successful is below `low_min_score_successful` (a weak lead).
  LOW     the model predicts "Successful" with a score of at least that value.

Risk score = 1 - score(Successful), shown for ranking only. EBM was trained with
balanced class weights, so its scores are NOT real-world failure probabilities.

Run:  python -m src.decision_support.risk
      -> reports/risk_rules.md: how the categories line up with actual outcomes,
         using the EBM's out-of-fold predictions on the TRAINING set (test set untouched).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_logger, load_config, resolve

log = get_logger("risk")
RISK_LEVELS = ["Low", "Medium", "High"]


def assign_risk(proba, class_order: list[str], min_success: float | None = None) -> dict:
    """Return risk level, risk score and a plain-language reason for one project."""
    proba = np.asarray(proba, dtype=float)
    if proba.shape != (len(class_order),) or not np.all(np.isfinite(proba)):
        raise ValueError(f"Expected {len(class_order)} finite class scores, got {proba}")
    if min_success is None:
        min_success = load_config()["risk"]["low_min_score_successful"]

    predicted = class_order[int(np.argmax(proba))]
    p_success = float(proba[class_order.index("Successful")])

    if predicted == "Unsuccessful":
        level, reason = "High", "The model predicts Unsuccessful."
    elif predicted == "Partially Successful":
        level, reason = "Medium", "The model predicts Partially Successful."
    elif p_success < min_success:
        level, reason = "Medium", (f"The model predicts Successful, but only with a weak lead "
                                   f"(score {p_success:.2f} < {min_success:.2f}).")
    else:
        level, reason = "Low", f"The model predicts Successful with a score of {p_success:.2f}."
    return {"risk_level": level, "risk_score": round(1 - p_success, 3),
            "predicted_outcome": predicted, "reason": reason}


def main() -> None:
    """Describe what each risk level meant on the training set (out-of-fold, unseen folds)."""
    cfg = load_config()
    order = cfg["target"]["class_order"]
    art = resolve(cfg["paths"]["artifacts_dir"])
    train = pd.read_csv(resolve(cfg["paths"]["processed_dir"]) / "train.csv")
    oof = np.load(art / "cv" / "EBM_oof.npy")

    levels = [assign_risk(p, order)["risk_level"] for p in oof]
    tab = pd.crosstab(pd.Categorical(levels, RISK_LEVELS), train[cfg["columns"]["target"]])[order]
    share = tab.div(tab.sum(axis=1), axis=0)

    lines = ["# Phase 6 — Risk rules", "",
             "| Risk level | Rule |", "|---|---|",
             "| **High** | Model predicts Unsuccessful |",
             "| **Medium** | Model predicts Partially Successful, or predicts Successful with score < "
             f"{cfg['risk']['low_min_score_successful']:.2f} |",
             f"| **Low** | Model predicts Successful with score ≥ {cfg['risk']['low_min_score_successful']:.2f} |", "",
             "The 0.50 cut-off is a **provisional, project-defined** choice. It means the model gives Successful "
             "at least half of its total score. It has not been validated against any external decision criterion.", "",
             "## What the levels looked like on the training set", "",
             "These come from the EBM's out-of-fold predictions: each project was scored by a model that "
             "had not seen it during training. The test set was not used.", "",
             "| Risk level | Projects | Actually Successful | Actually Partially | Actually Unsuccessful |",
             "|---|---|---|---|---|",
             *[f"| {lvl} | {int(tab.loc[lvl].sum()):,} | {share.loc[lvl, order[0]]:.0%} | "
               f"{share.loc[lvl, order[1]]:.0%} | {share.loc[lvl, order[2]]:.0%} |" for lvl in RISK_LEVELS], "",
             "How to read this:",
             f"- **Low** projects were mostly Successful ({share.loc['Low', order[0]]:.0%}).",
             f"- **High** projects had the largest Unsuccessful share ({share.loc['High', order[2]]:.0%}, against "
             f"{share.loc[['Low', 'Medium'], order[2]].min():.0%}–{share.loc[['Low', 'Medium'], order[2]].max():.0%} elsewhere). But "
             f"{share.loc['High', order[0]]:.0%} of them still turned out Successful. The model over-flags on purpose "
             "(balanced class weights), so High means **'review carefully'**, not 'will fail'.",
             "- **Medium** projects were mostly Partially Successful or Successful."]
    (resolve(cfg["paths"]["reports_dir"]) / "risk_rules.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
