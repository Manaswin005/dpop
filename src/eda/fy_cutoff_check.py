"""Phase 3 pre-check: does the outcome scale look different before FY1990?

Run:  python -m src.eda.fy_cutoff_check

Descriptive only (training set). It does NOT filter the dataset.
Writes reports/fy1990_check.md.
"""
from __future__ import annotations

import pandas as pd

from src.preprocessing.prepare import load_split
from src.utils.config import get_logger, load_config, resolve

log = get_logger("fy_check")
CUTOFF = 1990


def main() -> None:
    cfg = load_config()
    order = cfg["target"]["class_order"]
    train, _ = load_split()
    period = train["Approval FY"].lt(CUTOFF).map({True: f"Before FY{CUTOFF}", False: f"FY{CUTOFF} onward"})

    counts = pd.crosstab(period, train["outcome_class"])[order]
    shares = counts.div(counts.sum(axis=1), axis=0)
    counts["Total"] = counts.sum(axis=1)
    cost_reported = train.groupby(period)["lending_cost_reported"].mean()

    lines = [f"# FY{CUTOFF} cutoff check (training set, descriptive only)", "",
             "| Period | " + " | ".join(order) + " | Total | Cost reported |",
             "|---|" + "---|" * (len(order) + 2)]
    for p in counts.index:
        cells = [f"{counts.loc[p, c]:,} ({shares.loc[p, c]:.0%})" for c in order]
        lines.append(f"| {p} | " + " | ".join(cells) + f" | {counts.loc[p, 'Total']:,} | {cost_reported[p]:.0%} |")
    lines += ["", "Decision: **no filter applied.** The full eligible dataset is kept. "
              "The difference before FY1990 is recorded as a limitation (see README)."]
    resolve(cfg["paths"]["reports_dir"]).joinpath("fy1990_check.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
