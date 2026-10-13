"""Stage 2 Exploratory Baseline Modeling (Isolated Experiment).

Fits an exploratory penalized logistic regression and decision tree classifier
on the 14-project supervision pilot to evaluate feasibility while preventing
target leakage and preserving all Stage 1 artifacts.

Features used (genuine interim signals):
- pdo_rating_numeric (ordinal encoding of task team rating)
- ip_rating_numeric (ordinal encoding of implementation progress)
- disbursement_percentage (cumulative funds drawn down)
- schedule_delay_months (delay beyond original closing date)
- stage1_risk_score (approval-time prior from EBM)
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

RATING_MAP = {
    "Highly Unsatisfactory": 1,
    "Unsatisfactory": 2,
    "Moderately Unsatisfactory": 3,
    "Moderately Satisfactory": 4,
    "Satisfactory": 5,
    "Highly Satisfactory": 6,
}


def prepare_stage2_data(
    snapshots_csv: str = "data/stage2/pilot_snapshots.csv",
    comparison_csv: str = "stage2_experiments/pilot_comparison_table.csv",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    snaps = pd.read_csv(snapshots_csv)
    comp = pd.read_csv(comparison_csv)

    merged = snaps.merge(
        comp[["project_id", "stage1_p_successful", "stage1_p_partially", "stage1_p_unsuccessful"]],
        on="project_id",
        how="left",
    )

    # Encode ordinal ratings (impute neutral median 4 if unstated)
    merged["pdo_num"] = merged["pdo_rating_current"].map(RATING_MAP).fillna(4.0)
    merged["ip_num"] = merged["ip_rating_current"].map(RATING_MAP).fillna(4.0)
    merged["disb_pct"] = merged["disbursement_percentage"].fillna(0.0)
    merged["delay_mo"] = merged["schedule_delay_months"].fillna(0.0)

    # Prior risk from Stage 1: P(Unsuccessful)
    merged["stage1_prior_unsuc"] = merged["stage1_p_unsuccessful"].fillna(0.20)

    features = ["pdo_num", "ip_num", "disb_pct", "delay_mo", "stage1_prior_unsuc"]
    X = merged[features]
    y = merged["outcome_class"]

    return X, y, features


def train_stage2_exploratory(output_dir: str = "stage2_experiments") -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X, y, features = prepare_stage2_data()

    # 1. Fit Penalized Logistic Regression (L2 / Ridge with strong regularization)
    clf_lr = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=500,
        random_state=42,
    )
    clf_lr.fit(X, y)

    # 2. Fit Constrained Decision Tree (max_depth=2 to prevent wild overfitting)
    clf_dt = DecisionTreeClassifier(
        max_depth=2,
        class_weight="balanced",
        random_state=42,
    )
    clf_dt.fit(X, y)

    # Save artifacts in isolated stage2_experiments directory
    joblib.dump(clf_lr, out_path / "stage2_logistic_exploratory.joblib")
    joblib.dump(clf_dt, out_path / "stage2_tree_exploratory.joblib")

    # Leave-one-out cross validation predictions
    loo_preds = []
    for i in range(len(X)):
        X_train = X.drop(index=i)
        y_train = y.drop(index=i)
        X_test = X.iloc[[i]]

        m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)[0]
        loo_preds.append(pred)

    loo_acc = float(accuracy_score(y, loo_preds))

    metrics = {
        "sample_size": len(X),
        "features": features,
        "class_distribution": y.value_counts().to_dict(),
        "train_accuracy_logistic": float(accuracy_score(y, clf_lr.predict(X))),
        "train_accuracy_tree": float(accuracy_score(y, clf_dt.predict(X))),
        "leave_one_out_accuracy_logistic": round(loo_acc, 3),
        "logistic_classes": list(clf_lr.classes_),
        "notes": "Exploratory model trained on 14 supervision pilot records. Not intended as a validated replacement for Stage 1.",
    }

    (out_path / "stage2_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    res = train_stage2_exploratory()
    print("Stage 2 exploratory training complete:")
    print(json.dumps(res, indent=2))
