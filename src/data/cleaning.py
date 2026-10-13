"""Stateless cleaning steps that are safe to run BEFORE the train-test split.

"Stateless" means each step uses only fixed rules written in config.yaml or
values within the same row/project. Nothing here is learned from the data
distribution (no means, medians, encoders), so no information leaks from
future test rows into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_logger, load_config

log = get_logger(__name__)

EXCEL_EPOCH = pd.Timestamp("1899-12-30")  # origin used by Excel serial dates


# ----------------------------------------------------------------------------
# Date helpers
# ----------------------------------------------------------------------------
def parse_mixed_date(series: pd.Series) -> pd.Series:
    """Parse dates stored either as 'M/D/YYYY' text or as Excel serial numbers.

    IEG_EvalDate contains both forms (e.g. '6/29/2018' and '43447').
    """
    s = series.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    is_serial = numeric.notna() & numeric.between(1, 80000)
    out = pd.to_datetime(s.where(~is_serial), format="%m/%d/%Y", errors="coerce")
    out[is_serial] = EXCEL_EPOCH + pd.to_timedelta(numeric[is_serial], unit="D")
    return out


def fiscal_year_from_date(dates: pd.Series) -> pd.Series:
    """World Bank fiscal year runs July–June: July 2005 belongs to FY2006."""
    return dates.dt.year + (dates.dt.month >= 7).astype(int)


# ----------------------------------------------------------------------------
# Target mapping
# ----------------------------------------------------------------------------
def map_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map IEG ratings to three classes and drop ineligible labels.

    Returns the filtered frame and a summary dict for documentation.
    """
    cfg = load_config()
    tcol, out_col = cfg["columns"]["target_raw"], cfg["columns"]["target"]
    mapping, excluded = cfg["target"]["mapping"], cfg["target"]["excluded_labels"]

    raw = df[tcol].astype(str).str.strip()
    unknown = sorted(set(raw) - set(mapping) - set(excluded))
    if unknown:
        raise ValueError(f"Unrecognised {tcol} labels (update config): {unknown}")

    excluded_counts = raw[raw.isin(excluded)].value_counts().to_dict()
    keep = raw.isin(mapping)
    out = df.loc[keep].copy()
    out[out_col] = raw[keep].map(mapping)
    summary = {
        "rows_in": int(len(df)),
        "rows_kept": int(keep.sum()),
        "excluded_counts": {k: int(v) for k, v in excluded_counts.items()},
    }
    log.info("Target mapping kept %s of %s rows", summary["rows_kept"], summary["rows_in"])
    return out, summary


# ----------------------------------------------------------------------------
# Repeated Project IDs
# ----------------------------------------------------------------------------
def select_one_evaluation_per_project(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one rated evaluation per project.

    Repeated IDs are separate IEG evaluations of the SAME project (e.g. an
    ICR Review 'ES' followed later by a field-based assessment 'PAR'). Their
    approval-time features are identical, so keeping both would duplicate the
    project and sometimes give it two contradictory labels.

    Rule: keep the most recent rated evaluation (the latest IEG judgement);
    ties on date are broken by evaluation-type priority from config.
    Must be called AFTER map_target so an unrated later review never displaces
    an earlier rated one.
    """
    cfg = load_config()
    pid, etype = cfg["columns"]["project_id"], cfg["columns"]["eval_type"]
    priority = {t: i for i, t in enumerate(cfg["record_selection"]["eval_type_priority"])}

    work = df.copy()
    work["_eval_date"] = parse_mixed_date(work[cfg["columns"]["eval_date"]])
    work["_type_rank"] = work[etype].map(priority).fillna(len(priority))
    # Sort so the preferred record is LAST within each project, then keep last.
    work = work.sort_values([pid, "_eval_date", "_type_rank"],
                            ascending=[True, True, False], na_position="first")
    selected = work.drop_duplicates(subset=pid, keep="last")

    dup_ids = df[pid][df[pid].duplicated(keep=False)].unique()
    summary = {
        "rows_in": int(len(df)),
        "unique_projects": int(df[pid].nunique()),
        "projects_with_multiple_rated_evaluations": int(len(dup_ids)),
        "rows_removed": int(len(df) - len(selected)),
    }
    return selected.drop(columns=["_eval_date", "_type_rank"]), summary


# ----------------------------------------------------------------------------
# Data-quality fixes
# ----------------------------------------------------------------------------
def flag_upstream_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Replace constant fill values from the upstream 'cleaning' with NaN.

    Adds `lending_cost_reported` (1 = genuine value, 0 = was filled upstream).
    The real imputation is then learned later on training data only.
    """
    cfg = load_config()
    out = df.copy()
    for col, constant in cfg["upstream_imputation"].items():
        filled = np.isclose(out[col].astype(float), float(constant))
        out[col] = out[col].where(~filled)
        if col == "Lending Project Cost":
            out["lending_cost_reported"] = (~filled).astype(int)
    return out


def fix_approval_fy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace impossible Approval FY values using the Approval Date."""
    cfg = load_config()
    lo, hi = cfg["valid_approval_fy"]
    out = df.copy()
    derived = fiscal_year_from_date(parse_mixed_date(out["Approval Date"]))
    invalid = ~out["Approval FY"].between(lo, hi)
    out.loc[invalid, "Approval FY"] = derived[invalid]
    mismatch = int(((derived != out["Approval FY"]) & ~invalid).sum())
    return out, {"invalid_replaced": int(invalid.sum()), "date_fy_mismatch_kept": mismatch}


def harmonise_region(df: pd.DataFrame) -> pd.DataFrame:
    """Map later Africa sub-region codes back to one consistent 'AFR' label."""
    out = df.copy()
    out["Region"] = out["Region"].replace(load_config()["region_harmonisation"])
    return out
