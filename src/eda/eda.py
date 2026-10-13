"""Phase 1 — Exploratory data analysis (architecture Stage 2).

Run:  python -m src.eda.eda

Uses the TRAINING set only, so nothing about the test set influences
later decisions. Saves charts to reports/figures/ and a short summary to
reports/eda_report.md.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only, works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.prepare import load_split
from src.utils.config import get_logger, load_config, resolve

log = get_logger("eda")

# One colour per outcome class, used in every chart (green / amber / red)
CLASS_COLORS = {"Successful": "#2E7D5B", "Partially Successful": "#D89B2B", "Unsuccessful": "#C0453A"}


def _save(fig, name: str) -> str:
    out = resolve(load_config()["paths"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return name


def stacked_share(df: pd.DataFrame, by: str, title: str, fname: str, top: int | None = None,
                  chronological: bool = False) -> pd.DataFrame:
    """Horizontal 100% stacked bars: outcome mix within each group."""
    order = load_config()["target"]["class_order"]
    tab = pd.crosstab(df[by], df["outcome_class"])[order]
    if top:
        tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index[:top]]
    share = tab.div(tab.sum(axis=1), axis=0)
    # Time groups read top-to-bottom oldest first; other groups sorted by success share
    share = share.sort_index(ascending=False) if chronological else share.sort_values("Successful")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(share) + 1.2))
    left = np.zeros(len(share))
    for cls in order:
        ax.barh(share.index.astype(str), share[cls], left=left, color=CLASS_COLORS[cls], label=cls)
        left += share[cls].to_numpy()
    for i, n in enumerate(tab.loc[share.index].sum(axis=1)):
        ax.text(1.01, i, f"n={n}", va="center", fontsize=8, color="#555")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Share of projects")
    ax.set_title(title, loc="left", fontsize=11, pad=24)
    ax.legend(ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, fname)
    return share


def main() -> None:
    cfg = load_config()
    train, _ = load_split()
    order = cfg["target"]["class_order"]
    lines = ["# Phase 1 — Exploratory Data Analysis (training set only)", "",
             f"Training projects: {len(train):,}. All figures are in `reports/figures/`.", ""]

    # 1. Target distribution
    counts = train["outcome_class"].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(order, counts, color=[CLASS_COLORS[c] for c in order])
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Outcome class distribution", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "eda_01_class_distribution.png")
    lines += ["## Class balance", "",
              *[f"- {c}: {v:,} ({v / counts.sum():.1%})" for c, v in counts.items()],
              "", "The classes are imbalanced (Unsuccessful is the smallest), so macro-F1 is used to compare models: "
              "it gives every class equal weight.", ""]

    # 2. Outcome by approval decade
    train["Approval decade"] = (train["Approval FY"] // 10 * 10).astype(int).astype(str) + "s"
    dec = stacked_share(train, "Approval decade", "Outcome mix by approval decade", "eda_02_outcome_by_decade.png",
                         chronological=True)
    dec = dec.sort_index()
    lines += ["## Time trend", "", "| Decade | Successful | Partially | Unsuccessful |", "|---|---|---|---|",
              *[f"| {d} | {r['Successful']:.0%} | {r['Partially Successful']:.0%} | {r['Unsuccessful']:.0%} |" for d, r in dec.iterrows()], ""]

    # 3. Region, sector, instrument
    reg = stacked_share(train, "Region", "Outcome mix by region", "eda_03_outcome_by_region.png")
    stacked_share(train, "Sector Board", "Outcome mix by sector (12 largest)", "eda_04_outcome_by_sector.png", top=12)
    ins = stacked_share(train, "Lending Instrument Type", "Outcome mix by instrument type", "eda_05_outcome_by_instrument.png")
    lines += ["## Region and instrument", "",
              f"- Highest success share by region: {reg['Successful'].idxmax()} ({reg['Successful'].max():.0%}); "
              f"lowest: {reg['Successful'].idxmin()} ({reg['Successful'].min():.0%}).",
              *[f"- {k}: {v:.0%} Successful" for k, v in ins["Successful"].items()], ""]

    # 4. Project cost (only where actually reported)
    rep = train[train["lending_cost_reported"] == 1]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = [np.log10(rep.loc[rep["outcome_class"] == c, "Lending Project Cost"]) for c in order]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, showfliers=False)
    ax.set_yticks(range(1, 4), order)
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.75)
    ax.set_xlabel("log10(Lending Project Cost, USD)")
    ax.set_title("Planned project cost by outcome (reported values only)", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "eda_06_cost_by_outcome.png")
    med = rep.groupby("outcome_class")["Lending Project Cost"].median().reindex(order)
    cost_flag = pd.crosstab(train["lending_cost_reported"], train["outcome_class"], normalize="index")[order]
    lines += ["## Project cost", "",
              *[f"- Median cost, {c}: ${v / 1e6:,.1f}M" for c, v in med.items()],
              f"- Success share when cost reported: {cost_flag.loc[1, 'Successful']:.0%}; when not reported: {cost_flag.loc[0, 'Successful']:.0%} "
              "(unreported cost is mostly older projects, so this partly reflects era).", ""]

    lines += ["## Notes", "", "- These are associations in historical data, not causes.",
              "- Feature selection (next phase) tests which of these features carry predictive signal."]
    resolve(cfg["paths"]["reports_dir"]).joinpath("eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("EDA complete: 6 figures + eda_report.md")


if __name__ == "__main__":
    main()
