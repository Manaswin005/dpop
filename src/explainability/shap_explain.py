"""Phase 5 — SHAP explanations for the selected model (architecture Stage 8).

Run:  python -m src.explainability.shap_explain

Method
- SHAP ExactExplainer (shap 0.51) on the model's class probabilities.
  There are only 7 features, so exact Shapley values are cheap: every
  combination of features is evaluated, no approximation.
- SHAP cannot mask text categories directly, so categories are passed to SHAP
  as integer codes and turned back into labels inside the prediction function.
  The saved model is used unchanged.
- Background (the "average project" SHAP compares against): 100 training rows.
- Explained rows: 500 test projects (stratified sample). Explaining predictions
  does not train or change anything.

Outputs
  reports/figures/shap_01_global_importance.png
  reports/figures/shap_02_local_example.png
  reports/shap_global_importance.csv
  reports/shap_explanation.md
  artifacts/shap/background.csv   (reused by the app later)
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.models.train import load_training_data
from src.utils.config import get_logger, load_config, resolve

log = get_logger("shap")
N_BACKGROUND, N_EXPLAIN = 100, 500
CLASS_COLORS = ["#2E7D5B", "#D89B2B", "#C0453A"]  # same colours as the EDA charts


# ----------------------------------------------------------------------------
# SHAP wrapper: integer codes in, labels to the model
# ----------------------------------------------------------------------------
class ShapModel:
    """Lets SHAP work with text categories by swapping them for integer codes."""

    def __init__(self, model, columns, cats, frames):
        self.model, self.columns, self.cats = model, list(columns), cats
        # Code table built from the values present (a lookup, not learned from outcomes)
        self.levels = {c: sorted(set().union(*[set(f[c]) for f in frames])) for c in cats}

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.columns].copy()
        for c in self.cats:
            out[c] = out[c].map({v: i for i, v in enumerate(self.levels[c])})
        return out.astype(float)

    def predict_proba(self, arr) -> np.ndarray:
        df = pd.DataFrame(np.asarray(arr), columns=self.columns)
        for c in self.cats:
            df[c] = [self.levels[c][int(i)] for i in df[c]]
        return self.model.predict_proba(df)


def explain(model, X_background, X_explain, cats, frames):
    sm = ShapModel(model, X_explain.columns, cats, frames)
    explainer = shap.Explainer(sm.predict_proba, shap.maskers.Independent(sm.encode(X_background)),
                               algorithm="exact")
    sv = explainer(sm.encode(X_explain), silent=True)
    return sv  # values: (rows, features, classes)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_global(importance: pd.DataFrame, order, path) -> None:
    """Mean |SHAP| per feature, one bar segment per outcome class."""
    imp = importance.sort_values("total")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = np.zeros(len(imp))
    for cls, col in zip(order, CLASS_COLORS):
        ax.barh(imp.index, imp[cls], left=left, color=col, label=cls)
        left += imp[cls].to_numpy()
    for i, v in enumerate(imp["total"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (average change in predicted probability)")
    ax.set_title("Global feature importance — EBM (500 test projects)", loc="left", fontsize=11, pad=22)
    ax.legend(ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, imp["total"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_local(sv, row: int, cls: int, display: pd.Series, order, path) -> None:
    """SHAP waterfall for one project and one class, with readable feature values."""
    exp = shap.Explanation(values=sv.values[row, :, cls], base_values=sv.base_values[row, cls],
                           data=display.to_numpy(), feature_names=list(display.index))
    shap.plots.waterfall(exp, show=False, max_display=8)
    fig = plt.gcf()
    fig.set_size_inches(9, 4.6)
    plt.title(f"Why this project got P({order[cls]}) = {exp.base_values + exp.values.sum():.2f}",
              loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _display_values(x: pd.Series) -> pd.Series:
    """Human-readable feature values for the waterfall labels."""
    out = x.astype(object).copy()
    cost = x["Lending Project Cost"]
    out["Lending Project Cost"] = "not reported" if pd.isna(cost) else f"${cost / 1e6:,.1f}M"
    out["Approval FY"] = int(x["Approval FY"])
    out["lending_cost_reported"] = "yes" if x["lending_cost_reported"] == 1 else "no"
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    art = resolve(cfg["paths"]["artifacts_dir"])
    figs = resolve(cfg["paths"]["figures_dir"])
    reports = resolve(cfg["paths"]["reports_dir"])

    meta = json.loads((art / "model_metadata.json").read_text())
    model = joblib.load(art / "models" / meta["selected_model_file"])
    X, y, X_test, y_test, cats, nums, order = load_training_data()

    background = X.sample(N_BACKGROUND, random_state=seed)
    idx, _ = train_test_split(np.arange(len(X_test)), train_size=N_EXPLAIN, stratify=y_test, random_state=seed)
    X_exp = X_test.iloc[np.sort(idx)]
    log.info("Computing exact SHAP values for %d test projects ...", len(X_exp))
    sv = explain(model, background, X_exp, cats, frames=[X, X_test])

    # Sanity check: base value + contributions must equal the model's probabilities
    err = np.abs(sv.base_values + sv.values.sum(axis=1) - model.predict_proba(X_exp)).max()
    log.info("Additivity check: max error %.2e", err)

    # Global importance: mean absolute SHAP value per feature and class
    imp = pd.DataFrame(np.abs(sv.values).mean(axis=0), index=X_exp.columns, columns=order)
    imp["total"] = imp.sum(axis=1)
    imp = imp.sort_values("total", ascending=False)
    imp.round(4).to_csv(reports / "shap_global_importance.csv")
    plot_global(imp, order, figs / "shap_01_global_importance.png")

    # Local example: first explained project the model predicts as Unsuccessful
    # (chosen by the prediction only, not by whether it turned out correct)
    proba = model.predict_proba(X_exp)
    unsucc = order.index("Unsuccessful")
    row = int(np.flatnonzero(proba.argmax(1) == unsucc)[0])
    display = _display_values(X_exp.iloc[row])
    plot_local(sv, row, unsucc, display, order, figs / "shap_02_local_example.png")
    contrib = pd.DataFrame({"value": display, "shap_P_unsuccessful": sv.values[row, :, unsucc].round(4)})
    contrib = contrib.reindex(contrib["shap_P_unsuccessful"].abs().sort_values(ascending=False).index)

    (art / "shap").mkdir(parents=True, exist_ok=True)
    background.to_csv(art / "shap" / "background.csv", index=False)
    _write_report(reports, imp, order, contrib, proba[row], order[y_test[X_exp.index[row]]],
                  sv.base_values[row], err, len(X_exp))
    log.info("SHAP outputs written (global + local example)")


def _write_report(reports, imp, order, contrib, p, true_label, base, err, n):
    top = imp.index[:3].tolist()
    lines = [
        "# Phase 5 — Explaining the EBM model with SHAP", "",
        f"Method: exact SHAP values on predicted class probabilities, {n} test projects, "
        f"100 training projects as background. Additivity check: max error {err:.1e} "
        "(contributions add up exactly to the model's probabilities).", "",
        "## 1. Global importance (`shap_01_global_importance.png`)", "",
        "| Feature | Successful | Partially Successful | Unsuccessful | Total |", "|---|---|---|---|---|",
        *[f"| {f} | {r[order[0]]:.3f} | {r[order[1]]:.3f} | {r[order[2]]:.3f} | {r['total']:.3f} |" for f, r in imp.iterrows()], "",
        "**How to read it:**",
        "- Each bar shows how much a feature moves the predicted probabilities, on average, across the 500 projects.",
        "- A value of 0.10 means the feature typically shifts a class probability by about 10 percentage points.",
        "- The coloured segments show which outcome class the shift affects.",
        f"- The most influential features are **{top[0]}**, **{top[1]}** and **{top[2]}**.", "",
        "## 2. One project explained (`shap_02_local_example.png`)", "",
        f"The model predicts this project as Unsuccessful with probability {p[order.index('Unsuccessful')]:.2f} "
        f"(true outcome: {true_label}). It starts from the average P(Unsuccessful) of {base[order.index('Unsuccessful')]:.2f}:", "",
        "| Feature | Value | Change in P(Unsuccessful) |", "|---|---|---|",
        *[f"| {f} | {r.value} | {r.shap_P_unsuccessful:+.3f} |" for f, r in contrib.iterrows()], "",
        "**How to read it:**",
        "- Start at the bottom value E[f(x)], the average prediction.",
        "- Red bars push the probability of Unsuccessful up; blue bars push it down.",
        "- The top value f(x) is this project's final probability.", "",
        "## 3. Cautions", "",
        "- SHAP shows what the model **relies on** to make predictions. It does **not** show causes. For example, an "
        "approval year that raises the predicted risk does not mean the year caused the outcome.",
        "- Approval FY, Lending Project Cost and lending_cost_reported are related (older projects rarely have a reported cost). "
        "The model's credit can be split between them, so read them together.",
        "- Approval FY partly reflects the change in rating scale before FY1990 (see README).",
        "- EBM was trained with balanced class weights, so its probabilities are tilted towards the smaller classes. "
        "That is why the average P(Unsuccessful) is above the 18.5% share in the data. These are model scores, "
        "not calibrated real-world probabilities.",
        "- Only approval-time features are used. No post-completion information is an input.",
    ]
    (reports / "shap_explanation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
