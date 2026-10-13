"""Phase 1–2: build the modelling dataset and the leakage-safe 80:20 split.

Run:  python -m src.preprocessing.prepare

Steps
1. Reuse the Phase 0 stateless cleaning (target mapping, one evaluation per
   project, fill-value repair, Approval FY fix, region harmonisation).
2. Keep only Project ID (for traceability), approval-time features and the target.
3. Stratified 80:20 split. Each project now has exactly one row, so a random
   split by row is also a split by project (checked by a test).
Nothing in this file learns from data values, so it is safe before splitting.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import cleaning
from src.data.load import load_raw
from src.utils.config import get_logger, load_config, resolve

log = get_logger("prepare")


def feature_lists(cfg: dict | None = None) -> tuple[list[str], list[str]]:
    """Categorical and numerical feature names (numerical includes derived flag)."""
    cfg = cfg or load_config()
    cats = list(cfg["features"]["categorical"])
    nums = list(cfg["features"]["numerical"]) + list(cfg["features"]["derived"])
    return cats, nums


def build_modelling_data() -> pd.DataFrame:
    """Raw CSV -> one row per project with approval-time features + target."""
    cfg = load_config()
    df = load_raw()
    df, _ = cleaning.map_target(df)
    df, _ = cleaning.select_one_evaluation_per_project(df)
    df, _ = cleaning.fix_approval_fy(df)
    df = cleaning.harmonise_region(cleaning.flag_upstream_imputation(df))
    cats, nums = feature_lists(cfg)
    keep = [cfg["columns"]["project_id"], *cats, *nums, cfg["columns"]["target"]]
    return df[keep].reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    train, test = train_test_split(
        df, test_size=cfg["split"]["test_size"], stratify=df[cfg["columns"]["target"]],
        random_state=cfg["project"]["random_seed"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved train/test files (used by every later phase)."""
    cfg = load_config()
    d = resolve(cfg["paths"]["processed_dir"])
    if not (d / "train.csv").exists():
        raise FileNotFoundError("Run `python -m src.preprocessing.prepare` first.")
    return pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")


def main() -> None:
    cfg = load_config()
    df = build_modelling_data()
    train, test = split_data(df)
    out = resolve(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    tgt = cfg["columns"]["target"]
    info = {
        "rows_total": len(df), "rows_train": len(train), "rows_test": len(test),
        "train_class_counts": train[tgt].value_counts().to_dict(),
        "test_class_counts": test[tgt].value_counts().to_dict(),
    }
    (out / "split_info.json").write_text(json.dumps(info, indent=2))
    log.info("Saved split: %s", info)


if __name__ == "__main__":
    main()
