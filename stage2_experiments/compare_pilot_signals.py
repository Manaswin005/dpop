"""Stage 2 Exploratory Feasibility & Descriptive Alignment.

Compares Stage 1 approval-time baseline predictions against interim supervision
signals (PDO/IP ratings, disbursement percentage) for the 12 pilot snapshots.
Does NOT train a statistical model due to small sample size and missing Unsuccessful class.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate_exploratory_comparison(
    snapshots_csv: str | Path = "data/stage2/pilot_snapshots.csv",
    train_csv: str | Path = "data/processed/train.csv",
    oof_npy: str | Path = "artifacts/cv/EBM_oof.npy",
    metadata_json: str | Path = "artifacts/model_metadata.json",
    output_dir: str | Path = "stage2_experiments",
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps = pd.read_csv(snapshots_csv)
    train = pd.read_csv(train_csv)
    oof = np.load(oof_npy)
    meta = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    classes = meta["class_order"]

    train_idx = {pid: i for i, pid in enumerate(train["Project ID"])}

    records = []
    for _, r in snaps.iterrows():
        pid = r["project_id"]
        idx = train_idx[pid]
        probs = oof[idx]
        stage1_pred = classes[np.argmax(probs)]
        stage1_conf = float(np.max(probs))

        records.append({
            "project_id": pid,
            "snapshot_date": r["snapshot_date"],
            "sequence": r["snapshot_sequence"],
            "pdo_rating_current": r["pdo_rating_current"],
            "ip_rating_current": r["ip_rating_current"],
            "disbursement_percentage": r["disbursement_percentage"],
            "schedule_delay_months": r["schedule_delay_months"],
            "actual_outcome": r["outcome_class"],
            "stage1_pred": stage1_pred,
            "stage1_p_successful": round(float(probs[0]), 3),
            "stage1_p_partially": round(float(probs[1]), 3),
            "stage1_p_unsuccessful": round(float(probs[2]), 3),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "pilot_comparison_table.csv", index=False)
    return df


if __name__ == "__main__":
    df = generate_exploratory_comparison()
    print("Exported pilot comparison table to stage2_experiments/pilot_comparison_table.csv")
    print(df[["project_id", "actual_outcome", "stage1_pred", "pdo_rating_current", "ip_rating_current", "disbursement_percentage"]])
