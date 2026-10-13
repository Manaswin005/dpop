"""Phase 2 — Feature selection (architecture Stage 3), TRAINING data only.

Run:  python -m src.features.selection

Three methods from the architecture, applied to each original feature:
  1. Correlation  - numeric Pearson correlation and categorical Cramér's V,
                    used to find redundant feature pairs.
  2. Information gain - mutual information between feature and outcome.
  3. Chi-square   - independence test between feature and outcome
                    (numeric features are binned into quintiles first).

Rules (thresholds in config.yaml):
  - Drop a feature if it is weak on BOTH information gain and chi-square.
  - If two categorical features are nearly interchangeable (Cramér's V above
    the threshold), keep the one with higher information gain.
Output: artifacts/selected_features.json, reports/feature_selection.md, a heatmap.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from src.preprocessing.prepare import feature_lists, load_split
from src.utils.config import get_logger, load_config, resolve

log = get_logger("selection")


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categorical variables, 0 (none) to 1 (identical)."""
    tab = pd.crosstab(a, b)
    chi2 = chi2_contingency(tab, correction=False)[0]
    n, k = tab.to_numpy().sum(), min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def _numeric_for_tests(df: pd.DataFrame, nums: list[str]) -> pd.DataFrame:
    """Numeric copies used only for scoring: log cost, missing cost -> -1 marker."""
    x = df[nums].copy()
    x["Lending Project Cost"] = np.log10(x["Lending Project Cost"]).fillna(-1)
    return x


def score_features(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    cats, nums = feature_lists(cfg)
    y = train[cfg["columns"]["target"]]
    num_x = _numeric_for_tests(train, nums)
    seed = cfg["project"]["random_seed"]

    rows = []
    for f in cats + nums:
        if f in cats:
            codes = train[f].astype("category").cat.codes.to_numpy().reshape(-1, 1)
            mi = mutual_info_classif(codes, y, discrete_features=True, random_state=seed)[0]
            p = chi2_contingency(pd.crosstab(train[f], y))[1]
        else:
            col = num_x[f]
            discrete = col.nunique() <= 2
            mi = mutual_info_classif(col.to_frame(), y, discrete_features=discrete, random_state=seed)[0]
            # Quintile bins learned on training data (edges deduplicated for ties)
            binned = col if discrete else pd.qcut(col, 5, duplicates="drop")
            p = chi2_contingency(pd.crosstab(binned, y))[1]
        rows.append({"feature": f, "type": "categorical" if f in cats else "numerical",
                     "n_levels": int(train[f].nunique()), "mutual_info": round(float(mi), 4),
                     "chi2_p_value": float(p)})
    scores = pd.DataFrame(rows).sort_values("mutual_info", ascending=False).reset_index(drop=True)

    # Correlation: Cramér's V among categoricals, Pearson among numerics
    assoc = pd.DataFrame(np.eye(len(cats)), index=cats, columns=cats)
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            assoc.loc[a, b] = assoc.loc[b, a] = cramers_v(train[a], train[b])
    return scores, assoc


def decide(scores: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    fs = load_config()["feature_selection"]
    scores = scores.copy()
    scores["decision"], scores["reason"] = "KEEP", ""
    weak = (scores["mutual_info"] < fs["mi_min"]) & (scores["chi2_p_value"] > fs["chi2_p_max"])
    scores.loc[weak, ["decision", "reason"]] = ["DROP", "weak on both information gain and chi-square"]

    mi = scores.set_index("feature")["mutual_info"]
    cats = list(assoc.index)
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            if assoc.loc[a, b] >= fs["redundancy_cramers_v"]:
                loser = a if mi[a] < mi[b] else b
                winner = b if loser == a else a
                scores.loc[scores["feature"] == loser, ["decision", "reason"]] = [
                    "DROP", f"redundant with {winner} (Cramér's V={assoc.loc[a, b]:.2f})"]
    return scores


def plot_association(assoc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(assoc.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(assoc)), assoc.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(assoc)), assoc.index, fontsize=8)
    for i in range(len(assoc)):
        for j in range(len(assoc)):
            v = assoc.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "#333")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cramér's V")
    ax.set_title("Association between categorical features (training set)", loc="left", fontsize=10)
    out = resolve(load_config()["paths"]["figures_dir"])
    fig.tight_layout()
    fig.savefig(out / "fs_01_categorical_association.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    train, _ = load_split()
    scores, assoc = score_features(train)
    result = decide(scores, assoc)
    plot_association(assoc)

    _, nums = feature_lists(cfg)
    # Pearson on real values only: missing cost stays missing (pairwise-complete rows)
    corr_df = train[nums].assign(**{"Lending Project Cost": np.log10(train["Lending Project Cost"])})
    num_corr = corr_df.corr().round(2)
    kept = result.loc[result["decision"] == "KEEP", "feature"].tolist()
    cats_all, _ = feature_lists(cfg)
    selected = {"categorical": [f for f in cats_all if f in kept],
                "numerical": [f for f in nums if f in kept]}

    art = resolve(cfg["paths"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    (art / "selected_features.json").write_text(json.dumps(selected, indent=2))
    result.to_csv(resolve(cfg["paths"]["reports_dir"]) / "feature_selection_scores.csv", index=False)

    md = ["# Phase 2 — Feature selection (training set only)", "",
          "| Feature | Type | Levels | Information gain (MI) | Chi-square p | Decision | Reason |",
          "|---|---|---|---|---|---|---|",
          *[f"| {r.feature} | {r.type} | {r.n_levels} | {r.mutual_info:.4f} | {r.chi2_p_value:.2e} | {r.decision} | {r.reason} |"
            for r in result.itertuples()], "",
          "**Numeric correlation (Pearson, training set; log10 cost, reported values only):**", "", num_corr.to_markdown(), "",
          f"**Selected:** {', '.join(kept)}.", "",
          "Note: high information gain for features with many levels (e.g. Country Name) is partly an artefact of "
          "the number of categories; models limit this by grouping rare categories."]
    (resolve(cfg["paths"]["reports_dir"]) / "feature_selection.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Selected %d of %d features: %s", len(kept), len(result), kept)


if __name__ == "__main__":
    main()
