"""Phase 7 — Read-only loaders for the dashboard.

Everything here reads files saved by Phases 1–6. Nothing is trained or
recalculated. Each loader returns None when a file is missing, so the app can
show a friendly message instead of crashing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, resolve

CFG = load_config()
REPORTS = resolve(CFG["paths"]["reports_dir"])
FIGURES = resolve(CFG["paths"]["figures_dir"])
ARTIFACTS = resolve(CFG["paths"]["artifacts_dir"])
PROCESSED = resolve(CFG["paths"]["processed_dir"])
CLASS_ORDER = CFG["target"]["class_order"]


def figure(name: str) -> Path | None:
    p = FIGURES / name
    return p if p.exists() else None


def read_text(name: str) -> str | None:
    p = REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def read_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def model_metadata() -> dict | None:
    p = ARTIFACTS / "model_metadata.json"
    return json.loads(p.read_text()) if p.exists() else None


def model_comparison() -> pd.DataFrame | None:
    return read_csv(REPORTS / "model_comparison.csv")


def shap_importance() -> pd.DataFrame | None:
    return read_csv(REPORTS / "shap_global_importance.csv", index_col=0)


def training_data() -> pd.DataFrame | None:
    """Historical (training) projects only — the same data the EDA used."""
    return read_csv(PROCESSED / "train.csv")


def outcome_mix(df: pd.DataFrame, by: str, min_projects: int = 1) -> pd.DataFrame:
    """Count and outcome shares per group, e.g. per country or sector."""
    tab = pd.crosstab(df[by], df[CFG["columns"]["target"]]).reindex(columns=CLASS_ORDER, fill_value=0)
    out = tab.div(tab.sum(axis=1), axis=0).round(3)
    out.insert(0, "Projects", tab.sum(axis=1))
    return out[out["Projects"] >= min_projects].sort_values("Projects", ascending=False)


def section(markdown: str | None, heading: str) -> str | None:
    """Return one '## heading' section from a saved Markdown report."""
    if not markdown or f"## {heading}" not in markdown:
        return None
    part = markdown.split(f"## {heading}", 1)[1]
    return part.split("\n## ", 1)[0].strip()


def performance_notes(comp: pd.DataFrame, meta: dict) -> list[str]:
    """Plain-language notes for the performance page, built from the saved results (never hard-coded)."""
    notes = []
    sel = meta["selected_model"]
    rows = comp.set_index("model")
    if sel in rows.index:
        for name, r in rows.iterrows():
            if name != sel and r["test_macro_f1"] > rows.loc[sel, "test_macro_f1"]:
                notes.append(f"{name} has a higher test macro-F1 than {sel} ({r['test_macro_f1']:.3f} vs "
                             f"{rows.loc[sel, 'test_macro_f1']:.3f}), but selection uses the training-set CV score, "
                             f"where {sel} leads ({rows.loc[sel, 'cv_macro_f1']:.3f} vs {r['cv_macro_f1']:.3f}).")
    full = comp["train_rows_used"].max()
    for _, r in comp[comp["train_rows_used"] < full].iterrows():
        notes.append(f"{r['model']} was trained on a {int(r['train_rows_used']):,}-row subsample.")
    for name in meta.get("missing_model_files", []):
        notes.append(f"{name} was not evaluated in the last run because its model file was missing.")
    return notes


def held_out_size() -> int | None:
    p = PROCESSED / "split_info.json"
    return json.loads(p.read_text())["rows_test"] if p.exists() else None
