"""Phase 0 — Dataset audit (architecture Stage 0).

Run from the project root:
    python -m src.data.audit

Writes to reports/data_audit/:
    audit_summary.json        machine-readable findings
    column_profile.csv        per-column types, missingness, fill-value spikes
    feature_eligibility.csv   decision + rationale for every column
    data_audit_report.md      human-readable report
"""
from __future__ import annotations

import json
import platform
import sys

import numpy as np
import pandas as pd

from src.data import cleaning
from src.data.feature_eligibility import eligibility_table, included_features
from src.data.load import load_raw
from src.utils.config import get_logger, load_config, resolve

log = get_logger("audit")
SPIKE_THRESHOLD = 0.20  # a single numeric value in >20% of rows suggests an upstream fill


# ----------------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------------
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        top = s.value_counts(dropna=True)
        top_share = float(top.iloc[0] / len(s)) if len(top) else np.nan
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique()),
            "top_value": top.index[0] if len(top) else None,
            "top_value_share": round(top_share, 4),
            "suspected_fill_value": bool(pd.api.types.is_numeric_dtype(s) and top_share > SPIKE_THRESHOLD),
        })
    return pd.DataFrame(rows)


def net_commitment_evidence(df: pd.DataFrame) -> dict:
    nc, cost = df["Net Commitment"], df["Lending Project Cost"]
    return {
        "negative_values": int((nc < 0).sum()),
        "zero_values": int((nc == 0).sum()),
        "share_below_project_cost": round(float((nc < cost).mean()), 4),
        "share_non_round_10k": round(float((nc % 10_000 != 0).mean()), 4),
        "project_cost_share_non_round_10k": round(float((cost % 10_000 != 0).mean()), 4),
    }


def err_appraisal_evidence(df: pd.DataFrame) -> dict:
    s = df["ERR at Appraisal"]
    top = s.value_counts()
    pid = load_config()["columns"]["project_id"]
    multi = df[df[pid].duplicated(keep=False)]
    differs = multi.groupby(pid)["ERR at Appraisal"].nunique().gt(1).sum()
    return {
        "reported_missing": int(s.isna().sum()),
        "most_common_value": float(top.index[0]),
        "most_common_share": round(float(top.iloc[0] / len(s)), 4),
        "second_value_share": round(float(top.iloc[1] / len(s)), 4),
        "projects_where_value_differs_between_evaluations": int(differs),
    }


def repeated_id_evidence(df: pd.DataFrame) -> dict:
    pid = load_config()["columns"]["project_id"]
    multi = df[df[pid].duplicated(keep=False)]
    grp = multi.groupby(pid)
    # Which columns actually differ inside repeated-ID groups?
    differing = {c: int(grp[c].apply(lambda s: s.astype(str).nunique() > 1).sum()) for c in df.columns}
    differing = {k: v for k, v in sorted(differing.items(), key=lambda x: -x[1]) if v > 0}
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "rows_with_repeated_id": int(len(multi)),
        "repeated_ids": int(multi[pid].nunique()),
        "group_sizes": {str(k): int(v) for k, v in multi[pid].value_counts().value_counts().items()},
        "eval_type_patterns_top": {k: int(v) for k, v in grp["IEG_EvalType"].agg(lambda s: "|".join(sorted(s))).value_counts().head(6).items()},
        "groups_with_conflicting_raw_outcome": int(grp["IEG_Outcome"].nunique().gt(1).sum()),
        "columns_differing_within_groups": differing,
    }


def redundancy_evidence(df: pd.DataFrame) -> dict:
    def one_to_one(a, b):
        return bool(df.groupby(a)[b].nunique().max() == 1 and df.groupby(b)[a].nunique().max() == 1)
    codes = df.groupby("Country Code")["Country Name"].unique()
    return {
        "Country Code <-> Country Name one-to-one": one_to_one("Country Code", "Country Name"),
        "Country codes shared by several countries": {k: list(v) for k, v in codes[codes.str.len() > 1].items()},
        "Product Line Code <-> Product Line one-to-one": one_to_one("Product Line Code", "Product Line"),
    }


def label_selection_sensitivity(mapped: pd.DataFrame) -> dict:
    """How different would labels be if the EARLIEST evaluation were kept instead?"""
    cfg = load_config()
    pid, tgt = cfg["columns"]["project_id"], cfg["columns"]["target"]
    work = mapped.assign(_d=cleaning.parse_mixed_date(mapped[cfg["columns"]["eval_date"]]))
    multi = work[work[pid].duplicated(keep=False)]
    latest, _ = cleaning.select_one_evaluation_per_project(multi)
    earliest = multi.sort_values([pid, "_d"]).drop_duplicates(pid, keep="first")
    joined = latest.set_index(pid)[tgt].to_frame("latest").join(earliest.set_index(pid)[tgt].rename("earliest"))
    return {
        "multi_evaluation_projects": int(len(joined)),
        "three_class_label_differs_latest_vs_earliest": int((joined["latest"] != joined["earliest"]).sum()),
        "unparsed_eval_dates": int(work["_d"].isna().sum()),
    }


# ----------------------------------------------------------------------------
# Main audit
# ----------------------------------------------------------------------------
def run_audit() -> dict:
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["audit_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    profile = column_profile(raw)

    mapped, target_summary = cleaning.map_target(raw)
    selected, selection_summary = cleaning.select_one_evaluation_per_project(mapped)
    fixed, fy_summary = cleaning.fix_approval_fy(selected)
    fixed = cleaning.harmonise_region(cleaning.flag_upstream_imputation(fixed))

    tgt = cfg["columns"]["target"]
    order = cfg["target"]["class_order"]
    dist = fixed[tgt].value_counts().reindex(order)
    elig = eligibility_table()

    summary = {
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "pandas": pd.__version__, "numpy": np.__version__},
        "dataset": {"rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
                    "total_missing_cells": int(raw.isna().sum().sum())},
        "raw_outcome_counts": {k: int(v) for k, v in raw["IEG_Outcome"].value_counts().items()},
        "target_mapping": target_summary,
        "repeated_project_ids": repeated_id_evidence(raw),
        "record_selection": selection_summary,
        "label_selection_sensitivity": label_selection_sensitivity(mapped),
        "approval_fy_repair": fy_summary,
        "suspected_upstream_fill_values": profile.loc[profile["suspected_fill_value"], ["column", "top_value", "top_value_share"]].to_dict("records"),
        "lending_cost_reported_share_after_selection": round(float(fixed["lending_cost_reported"].mean()), 4),
        "net_commitment_evidence": net_commitment_evidence(raw),
        "err_at_appraisal_evidence": err_appraisal_evidence(raw),
        "redundancy": redundancy_evidence(raw),
        "region_counts_after_harmonisation": {k: int(v) for k, v in fixed["Region"].value_counts().items()},
        "modelling_dataset": {"rows": int(len(fixed)),
                              "class_counts": {k: int(v) for k, v in dist.items()},
                              "class_shares": {k: round(float(v / dist.sum()), 4) for k, v in dist.items()},
                              "approval_fy_range": [int(fixed["Approval FY"].min()), int(fixed["Approval FY"].max())]},
        "included_features": included_features(),
        "eligibility_decision_counts": elig["decision"].value_counts().to_dict(),
    }

    profile.to_csv(out_dir / "column_profile.csv", index=False)
    elig.to_csv(out_dir / "feature_eligibility.csv", index=False)
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    (out_dir / "data_audit_report.md").write_text(render_report(summary, elig), encoding="utf-8")
    log.info("Audit written to %s", out_dir)
    return summary


def render_report(s: dict, elig: pd.DataFrame) -> str:
    """Plain-language Markdown report built only from computed values."""
    t, r, sel, md = s["target_mapping"], s["repeated_project_ids"], s["record_selection"], s["modelling_dataset"]
    nc, err, sens = s["net_commitment_evidence"], s["err_at_appraisal_evidence"], s["label_selection_sensitivity"]
    lines = [
        "# Phase 0 — Dataset Audit Report", "",
        f"Source file: `{load_config()['paths']['raw_data']}` — {s['dataset']['rows']:,} rows × {s['dataset']['columns']} columns. "
        f"Missing cells as supplied: {s['dataset']['total_missing_cells']}.", "",
        "## 1. Target mapping",
        "Project-specific three-class grouping of IEG outcome ratings (not an official IEG convention):", "",
        "| IEG rating | Class |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in load_config()["target"]["mapping"].items()], "",
        f"Excluded labels: {t['excluded_counts']} → {t['rows_in'] - t['rows_kept']} rows removed; {t['rows_kept']:,} rows kept.", "",
        "## 2. Repeated Project IDs",
        f"- Exact duplicate rows: {r['exact_duplicate_rows']}.",
        f"- {r['repeated_ids']:,} Project IDs appear more than once ({r['rows_with_repeated_id']:,} rows); group sizes {r['group_sizes']}.",
        f"- Most common patterns: {r['eval_type_patterns_top']}.",
        "- Only evaluation fields differ within these groups; approval-time fields are identical. "
        "These are **separate IEG evaluations of the same project**, not separate projects.",
        f"- {r['groups_with_conflicting_raw_outcome']} groups carry different raw outcome ratings.", "",
        "**Decision:** keep one rated evaluation per project — the most recent (latest IEG judgement), "
        "tie-broken by evaluation-type priority (PAR first). "
        f"Rows removed: {sel['rows_removed']:,}. Sensitivity: choosing the earliest evaluation instead would change the "
        f"three-class label for {sens['three_class_label_differs_latest_vs_earliest']} of {sens['multi_evaluation_projects']:,} multi-evaluation projects. "
        "Train/test splitting will still be grouped by Project ID as a safeguard.", "",
        "## 3. Upstream fill values (data-quality finding)",
        "The supplied file reports zero missing values, but several numeric columns hold one constant in a large share of rows:", "",
        "| Column | Repeated value | Share of rows |", "|---|---|---|",
        *[f"| {x['column']} | {x['top_value']} | {x['top_value_share']:.1%} |" for x in s["suspected_upstream_fill_values"]], "",
        "These are median fills applied before the file was supplied. Treating them as real values would mislead the models.", "",
        "## 4. Key eligibility decisions",
        f"**Net Commitment — EXCLUDED.** {nc['negative_values']} negative values, {nc['share_below_project_cost']:.0%} below project cost, "
        f"{nc['share_non_round_10k']:.1%} non-round amounts (vs {nc['project_cost_share_non_round_10k']:.1%} for project cost). "
        "This pattern indicates amounts net of later cancellations; approval-time availability cannot be established.", "",
        f"**ERR at Appraisal — EXCLUDED from the primary model.** {err['most_common_share']:.1%} of rows equal {err['most_common_value']} "
        f"(next most common value: {err['second_value_share']:.1%}), so the true missingness is hidden. Values also differ between evaluations "
        f"of the same project in {err['projects_where_value_differs_between_evaluations']} projects, showing they were transcribed from evaluation records. "
        "Genuine appraisal values cannot be separated from filled ones.", "",
        f"**Lending Project Cost — INCLUDED with repair.** The fill value is converted back to missing and an indicator "
        f"`lending_cost_reported` is added; {s['lending_cost_reported_share_after_selection']:.1%} of modelling rows have a reported cost. "
        "Imputation will be fitted on training data only.", "",
        f"**Approval FY:** {s['approval_fy_repair']['invalid_replaced']} invalid value repaired from Approval Date; "
        f"{s['approval_fy_repair']['date_fy_mismatch_kept']} minor date/FY mismatches left as recorded.", "",
        f"**Region:** AFE/AFW harmonised to AFR. Counts: {s['region_counts_after_harmonisation']}.", "",
        f"**Redundant pairs:** {s['redundancy']}", "",
        "## 5. Modelling dataset after Phase 0 rules",
        f"{md['rows']:,} projects, approval FY {md['approval_fy_range'][0]}–{md['approval_fy_range'][1]}.", "",
        "| Class | Count | Share |", "|---|---|---|",
        *[f"| {k} | {md['class_counts'][k]:,} | {md['class_shares'][k]:.1%} |" for k in md["class_counts"]], "",
        f"Included approval-time features: {', '.join(s['included_features'])} (+ derived `lending_cost_reported`).", "",
        "## 6. Feature-eligibility register", "",
        "| Column | Decision | Rationale | Timing evidence |", "|---|---|---|---|",
        *[f"| {row.column} | {row.decision} | {row.rationale} | {row.timing_evidence} |" for row in elig.itertuples()], "",
        f"_Environment: Python {s['environment']['python']}, pandas {s['environment']['pandas']}, {s['environment']['platform']}._",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run_audit()
