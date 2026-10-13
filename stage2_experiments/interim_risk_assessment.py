"""Phase 7 - Minimal Stage 2 Interim Risk Assessment Demonstration.

Illustrates how genuine interim supervision signals (PDO/IP ratings, disbursement)
can provide an updated implementation-stage risk perspective alongside Stage 1.

NOTE: This is an illustrative, rule-based demonstration. It does NOT fit a machine-learning
classifier, does NOT claim predictive accuracy or validated failure detection,
and uses transparent, configurable risk rules.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def assess_interim_risk(
    pdo_rating: str | None,
    ip_rating: str | None,
    disbursement_pct: float | None,
    delay_months: float | None,
    stage1_risk: str,
) -> dict[str, str]:
    """Illustrative, configurable rule-based interim assessment.

    Labels are illustrative flags, NOT validated statistical failure predictions.
    """
    flags = []

    # 1. Rating signals (documented task-team ratings)
    is_pdo_distress = pdo_rating in ["Moderately Unsatisfactory", "Unsatisfactory", "Highly Unsatisfactory"]
    is_ip_distress = ip_rating in ["Moderately Unsatisfactory", "Unsatisfactory", "Highly Unsatisfactory"]
    is_mod_sat = pdo_rating == "Moderately Satisfactory" or ip_rating == "Moderately Satisfactory"

    if is_pdo_distress:
        flags.append(f"PDO rating downgraded to {pdo_rating}")
    if is_ip_distress:
        flags.append(f"Implementation Progress downgraded to {ip_rating}")

    # 2. Financial disbursement signals (documented ledger drawdown)
    is_disb_lag = pd.notna(disbursement_pct) and disbursement_pct <= 5.0
    if is_disb_lag:
        flags.append(f"Disbursement low at interim stage ({disbursement_pct:.1f}%)")

    # 3. Schedule delay signals
    has_delay = pd.notna(delay_months) and delay_months > 0.0
    if has_delay:
        flags.append(f"Closing date extended by {delay_months:.0f} months")

    # Illustrative combined stance
    if is_pdo_distress or (is_ip_distress and is_disb_lag):
        interim_stance = "High Implementation Alert"
    elif is_ip_distress or is_disb_lag or has_delay or is_mod_sat:
        interim_stance = "Moderate Implementation Attention"
    else:
        interim_stance = "Standard Supervision"

    return {
        "interim_risk_stance": interim_stance,
        "interim_signals_summary": "; ".join(flags) if flags else "No adverse flags identified",
    }


def run_demonstration(
    snapshots_csv: str | Path = "data/stage2/pilot_snapshots.csv",
    train_csv: str | Path = "data/processed/train.csv",
    oof_npy: str | Path = "artifacts/cv/EBM_oof.npy",
    metadata_json: str | Path = "artifacts/model_metadata.json",
    output_dir: str | Path = "stage2_experiments",
) -> pd.DataFrame:
    snaps = pd.read_csv(snapshots_csv)
    train = pd.read_csv(train_csv)
    oof = np.load(oof_npy)
    meta = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    classes = meta["class_order"]

    train_idx = {pid: i for i, pid in enumerate(train["Project ID"])}

    results = []
    for _, r in snaps.iterrows():
        pid = r["project_id"]
        idx = train_idx[pid]
        probs = oof[idx]

        # Stage 1 baseline risk assignment (from Phase 1 rules: score(Successful) >= 0.50 -> Low, etc.)
        p_succ = float(probs[0])
        p_unsuc = float(probs[2])
        if classes[np.argmax(probs)] == "Unsuccessful" or p_unsuc > 0.35:
            stage1_risk = "High"
        elif p_succ >= 0.50:
            stage1_risk = "Low"
        else:
            stage1_risk = "Medium"

        assessment = assess_interim_risk(
            pdo_rating=r["pdo_rating_current"] if pd.notna(r["pdo_rating_current"]) else None,
            ip_rating=r["ip_rating_current"] if pd.notna(r["ip_rating_current"]) else None,
            disbursement_pct=r["disbursement_percentage"] if pd.notna(r["disbursement_percentage"]) else None,
            delay_months=r["schedule_delay_months"] if pd.notna(r["schedule_delay_months"]) else None,
            stage1_risk=stage1_risk,
        )

        results.append({
            "project_id": pid,
            "snapshot_date": r["snapshot_date"],
            "sequence": r["snapshot_sequence"],
            "pdo_rating": r["pdo_rating_current"],
            "ip_rating": r["ip_rating_current"],
            "disbursement_pct": r["disbursement_percentage"],
            "schedule_delay_months": r["schedule_delay_months"],
            "stage1_baseline_risk": stage1_risk,
            "interim_risk_stance": assessment["interim_risk_stance"],
            "interim_signals": assessment["interim_signals_summary"],
            "eventual_outcome_target": r["outcome_class"],
        })

    df = pd.DataFrame(results)
    out_path = Path(output_dir) / "stage2_interim_assessment_demo.csv"
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = run_demonstration()
    print("Demonstration executed. Shape:", df.shape)
    print(df[["project_id", "stage1_baseline_risk", "interim_risk_stance", "eventual_outcome_target"]])
