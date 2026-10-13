"""Phases 3–4 — Train models with 5-fold CV, select the best, evaluate on test.

Run (from the project root):
    python -m src.models.train train              # all five models + ensemble
    python -m src.models.train train --models EBM QDA   # a subset
    python -m src.models.train summary            # Phase 3 training table
    python -m src.models.train evaluate           # selection + held-out test report

`train` only ever touches the TRAINING set. It saves, per model:
    artifacts/models/<name>.joblib   final model fitted on all training rows
    artifacts/cv/<name>.json         CV macro-F1 per fold, timing, status
    artifacts/cv/<name>_oof.npy      out-of-fold probabilities (used for the
                                     ensemble CV score and risk thresholds)
`evaluate` picks the winner by CV macro-F1 FIRST, then scores every trained
model once on the untouched test set. Test scores never change the winner.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import compute_metrics, plot_confusion_grid, report_text
from src.models.registry import MODEL_NAMES, AveragedEnsemble, ModelUnavailable, build_model
from src.preprocessing.prepare import load_split
from src.utils.config import get_logger, load_config, resolve

log = get_logger("train")
ENSEMBLE = "Ensemble (soft voting)"
# GPC is excluded from the ensemble: it is trained on a subsample, so its
# probabilities come from a different training set than the other members.
ENSEMBLE_MEMBERS = ["EBM", "RuleFit", "QDA", "TabPFN"]
# reg_param must be > 0: multiclass target encoding gives per-class columns that
# sum to 1 for each category, so unregularised covariance matrices are singular.
QDA_REG_GRID = [0.01, 0.05, 0.1, 0.3, 0.6]


# ----------------------------------------------------------------------------
# Shared data loading
# ----------------------------------------------------------------------------
def load_training_data():
    cfg = load_config()
    train, test = load_split()
    sel = json.loads((resolve(cfg["paths"]["artifacts_dir"]) / "selected_features.json").read_text())
    cats, nums = sel["categorical"], sel["numerical"]
    order = cfg["target"]["class_order"]
    to_code = {c: i for i, c in enumerate(order)}   # fixed class order everywhere
    tgt = cfg["columns"]["target"]
    return (train[cats + nums], train[tgt].map(to_code).to_numpy(),
            test[cats + nums], test[tgt].map(to_code).to_numpy(), cats, nums, order)


def folds(y):
    cfg = load_config()
    skf = StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["project"]["random_seed"])
    return list(skf.split(np.zeros(len(y)), y))


def _dirs():
    art = resolve(load_config()["paths"]["artifacts_dir"])
    for d in ("models", "cv"):
        (art / d).mkdir(parents=True, exist_ok=True)
    return art / "models", art / "cv"


# ----------------------------------------------------------------------------
# Cross-validation on the training set
# ----------------------------------------------------------------------------
def cross_validate(model, X, y, splits, n_classes):
    """Return out-of-fold probabilities and macro-F1 per fold."""
    oof = np.zeros((len(y), n_classes))
    scores = []
    for tr, va in splits:
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])
        scores.append(f1_score(y[va], oof[va].argmax(1), average="macro"))
    return oof, scores


def tune_qda(X, y, splits, cats, nums, n_classes):
    """Tiny grid for QDA covariance regularisation, scored by CV macro-F1."""
    results = {}
    for reg in QDA_REG_GRID:
        m = build_model("QDA", cats, nums).set_params(clf__reg_param=reg)
        try:
            _, s = cross_validate(m, X, y, splits, n_classes)
            results[reg] = float(np.mean(s))
        except np.linalg.LinAlgError:
            results[reg] = float("-inf")  # covariance still singular: unusable
    best = max(results, key=results.get)
    log.info("QDA reg_param grid (CV macro-F1): %s -> best %s", results, best)
    return best, results


def train_models(names: list[str]) -> None:
    X, y, _, _, cats, nums, order = load_training_data()
    splits = folds(y)
    model_dir, cv_dir = _dirs()

    for name in names:
        record = {"model": name, "status": "TRAINED", "n_train_rows": int(len(y))}
        try:
            model = build_model(name, cats, nums)
            if name == "QDA":
                best, grid = tune_qda(X, y, splits, cats, nums, len(order))
                model.set_params(clf__reg_param=best)
                record["qda_reg_param_grid"] = {str(k): v for k, v in grid.items()}
                record["qda_reg_param"] = best
            log.info("Cross-validating %s ...", name)
            t0 = time.time()
            oof, scores = cross_validate(model, X, y, splits, len(order))
            record["cv_seconds"] = round(time.time() - t0, 1)
            t0 = time.time()
            final = clone(model).fit(X, y)
            record["final_fit_seconds"] = round(time.time() - t0, 1)
            if name == "GPC":
                record["n_train_rows"] = int(final.n_train_used_)
                record["note"] = "Trained on a stratified subsample of the training set (GPC cost grows with n^3)."
            if name == "RuleFit":
                record["note"] = "imodels RuleFit is binary-only: three one-vs-rest RuleFit models, probabilities normalised."
            record["cv_macro_f1_folds"] = [round(float(s), 4) for s in scores]
            record["cv_macro_f1_mean"] = round(float(np.mean(scores)), 4)
            record["cv_macro_f1_std"] = round(float(np.std(scores)), 4)
            np.save(cv_dir / f"{name}_oof.npy", oof)
            joblib.dump(final, model_dir / f"{name}.joblib")
            log.info("%s CV macro-F1 = %.4f ± %.4f", name, record["cv_macro_f1_mean"], record["cv_macro_f1_std"])
        except ModelUnavailable as e:
            record.update(status="BLOCKED", reason=str(e))
            log.warning("%s blocked: %s", name, e)
        (cv_dir / f"{name}.json").write_text(json.dumps(record, indent=2))


def _ensemble(y, splits, available: dict):
    """Ensemble CV score = macro-F1 of averaged member OOF probabilities, per fold."""
    _, cv_dir = _dirs()
    members = [m for m in ENSEMBLE_MEMBERS if m in available]
    if len(members) < 2:
        return None, None, members
    oof = np.mean([np.load(cv_dir / f"{m}_oof.npy") for m in members], axis=0)
    scores = [f1_score(y[va], oof[va].argmax(1), average="macro") for _, va in splits]
    return oof, scores, members


def write_training_summary() -> None:
    """Phase 3 report: CV results, runtime and status per model (training set only)."""
    _, cv_dir = _dirs()
    lines = ["# Phase 3 — Training summary (5-fold CV on the training set only)", "",
             "| Model | Status | CV macro-F1 (mean ± std) | Training rows | CV time (s) | Final fit (s) |",
             "|---|---|---|---|---|---|"]
    notes = []
    for name in MODEL_NAMES:
        p = cv_dir / f"{name}.json"
        r = json.loads(p.read_text()) if p.exists() else {"status": "NOT RUN"}
        if r["status"] == "TRAINED":
            lines.append(f"| {name} | TRAINED | {r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f} | "
                         f"{r['n_train_rows']:,} | {r['cv_seconds']} | {r['final_fit_seconds']} |")
        else:
            lines.append(f"| {name} | {r['status']} | — | — | — | — |")
        if r.get("note"):
            notes.append(f"- {name}: {r['note']}")
        if r.get("reason"):
            notes.append(f"- {name} blocker: {r['reason']}")
        if r.get("qda_reg_param") is not None:
            notes.append(f"- QDA reg_param grid (CV macro-F1): {r['qda_reg_param_grid']} -> chosen {r['qda_reg_param']}")
    lines += ["", "Notes:", *notes, "",
              "No model has been scored on the test set yet (that is Phase 4)."]
    out = resolve(load_config()["paths"]["reports_dir"]) / "phase3_training_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ----------------------------------------------------------------------------
# Missing model files (e.g. GPC.joblib left out of a zip because of its size)
# ----------------------------------------------------------------------------
def find_missing_artifacts(records: dict, model_dir) -> dict:
    """Models whose training finished (status TRAINED) but whose .joblib file is absent."""
    return {n: model_dir / f"{n}.joblib" for n, r in records.items()
            if r["status"] == "TRAINED" and not (model_dir / f"{n}.joblib").exists()}


def missing_artifact_message(name: str, path) -> str:
    return (f"Model file for {name} not found: {path}. {name} is skipped in this evaluation. "
            f"To include it, copy the original {path.name} back into artifacts/models/, or rebuild it with "
            f"`python -m src.models.train train --models {name}` and then run evaluate again.")


# ----------------------------------------------------------------------------
# Selection + held-out test evaluation
# ----------------------------------------------------------------------------
def evaluate() -> None:
    cfg = load_config()
    X, y, X_test, y_test, cats, nums, order = load_training_data()
    splits = folds(y)
    model_dir, cv_dir = _dirs()
    reports = resolve(cfg["paths"]["reports_dir"])
    figs = resolve(cfg["paths"]["figures_dir"])

    records = {}
    for name in MODEL_NAMES:
        p = cv_dir / f"{name}.json"
        records[name] = json.loads(p.read_text()) if p.exists() else {"model": name, "status": "NOT RUN"}
    missing = find_missing_artifacts(records, model_dir)
    for name, path in missing.items():
        log.warning(missing_artifact_message(name, path))
    trained = {n: joblib.load(model_dir / f"{n}.joblib") for n, r in records.items()
               if r["status"] == "TRAINED" and n not in missing}

    # Ensemble: its CV score comes from saved out-of-fold files, so it is computed
    # exactly as before. The ensemble model itself is only rebuilt when every member
    # file is present; otherwise the existing Ensemble.joblib is used if available.
    trained_names = {n for n, r in records.items() if r["status"] == "TRAINED"}
    oof_e, scores_e, members = _ensemble(y, splits, trained_names)
    if scores_e is not None:
        if all(m in trained for m in members):
            ens = AveragedEnsemble({m: trained[m] for m in members}).fit()
            joblib.dump(ens, model_dir / "Ensemble.joblib")
            trained[ENSEMBLE] = ens
        elif (model_dir / "Ensemble.joblib").exists():
            trained[ENSEMBLE] = joblib.load(model_dir / "Ensemble.joblib")
        else:
            missing[ENSEMBLE] = model_dir / "Ensemble.joblib"
            log.warning(missing_artifact_message(ENSEMBLE, missing[ENSEMBLE]))
        np.save(cv_dir / "Ensemble_oof.npy", oof_e)
        records[ENSEMBLE] = {"model": ENSEMBLE, "status": "TRAINED", "members": members,
                             "n_train_rows": int(len(y)),
                             "cv_macro_f1_folds": [round(float(s), 4) for s in scores_e],
                             "cv_macro_f1_mean": round(float(np.mean(scores_e)), 4),
                             "cv_macro_f1_std": round(float(np.std(scores_e)), 4),
                             "note": f"Equal-weight average of {', '.join(members)} probabilities."}

    # 1) Choose the winner from CV only
    cv_scores = {n: r["cv_macro_f1_mean"] for n, r in records.items() if r["status"] == "TRAINED"}
    winner = max(cv_scores, key=cv_scores.get)
    log.info("Selected by CV macro-F1: %s (%.4f)", winner, cv_scores[winner])
    if winner in missing:
        raise FileNotFoundError(
            f"The CV-selected model {winner} cannot be evaluated because its file is missing. "
            + missing_artifact_message(winner, missing[winner]) + " No reports were changed.")

    # 2) Score every trained model once on the test set
    probas, rows, text = {}, [], []
    for name, model in trained.items():
        t0 = time.time()
        probas[name] = model.predict_proba(X_test)
        infer = time.time() - t0
        m = compute_metrics(y_test, probas[name], order)
        r = records[name]
        rows.append({"model": name, "cv_macro_f1": r["cv_macro_f1_mean"], "cv_std": r["cv_macro_f1_std"],
                     **{f"test_{k}": v for k, v in m.items()},
                     "train_rows_used": r.get("n_train_rows"), "cv_seconds": r.get("cv_seconds"),
                     "test_inference_seconds": round(infer, 2), "selected": name == winner})
        text += [f"## {name}", "", "```", report_text(y_test, probas[name], order), "```", ""]
    table = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    table.to_csv(reports / "model_comparison.csv", index=False)
    plot_confusion_grid(probas, y_test, order, figs / "eval_confusion_matrices.png")
    _plot_comparison(table, figs / "eval_model_comparison.png")

    blocked = [f"- **{n}**: {r['status']} — {r.get('reason', '')}" for n, r in records.items() if r["status"] != "TRAINED"]
    show = table[["model", "cv_macro_f1", "cv_std", "test_macro_f1", "test_accuracy", "test_weighted_f1",
                  "test_roc_auc_ovr_macro", "train_rows_used", "selected"]]
    md = ["# Model comparison", "",
          "Winner chosen by **training-set 5-fold CV macro-F1**. Test metrics are reported afterwards and do not change the choice.", "",
          show.to_markdown(index=False), "",
          *(["**Not trained:**", *blocked, ""] if blocked else []),
          *(["**Not evaluated in this run (model file missing):**",
             *[f"- **{n}**: `{p.name}` not found in artifacts/models/. Restore it or rebuild with "
               f"`python -m src.models.train train --models {n}`." for n, p in missing.items()], ""] if missing else []),
          "**Notes:**", *[f"- {n}: {r['note']}" for n, r in records.items() if r.get("note")],
          "", "# Classification reports (test set)", "", *text]
    (reports / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    meta = {"selected_model": winner, "selected_model_file": "Ensemble.joblib" if winner == ENSEMBLE else f"{winner}.joblib",
            "selection_metric": "cv_macro_f1_mean", "cv_scores": cv_scores, "class_order": order,
            "features": {"categorical": cats, "numerical": nums}, "random_seed": cfg["project"]["random_seed"],
            "model_status": {n: r["status"] for n, r in records.items()},
            "missing_model_files": sorted(missing),
            "versions": {"python": sys.version.split()[0], "platform": platform.platform(), "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__}}
    (resolve(cfg["paths"]["artifacts_dir"]) / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Evaluation written to reports/model_comparison.md")


def _plot_comparison(table: pd.DataFrame, path) -> None:
    t = table.sort_values("cv_macro_f1")
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(t) + 1.2))
    yy = np.arange(len(t))
    ax.barh(yy - 0.18, t["cv_macro_f1"], height=0.36, color="#3B6EA8", label="CV macro-F1 (train)")
    ax.barh(yy + 0.18, t["test_macro_f1"], height=0.36, color="#A9C4E2", label="Test macro-F1")
    ax.set_yticks(yy, [f"{m}  ★" if s else m for m, s in zip(t["model"], t["selected"])], fontsize=9)
    for i, (a, b) in enumerate(zip(t["cv_macro_f1"], t["test_macro_f1"])):
        ax.text(a + 0.005, i - 0.18, f"{a:.3f}", va="center", fontsize=7)
        ax.text(b + 0.005, i + 0.18, f"{b:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, max(t["cv_macro_f1"].max(), t["test_macro_f1"].max()) * 1.15)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Model comparison (★ = selected by CV)", loc="left", fontsize=11, pad=22)
    ax.legend(ncol=2, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "summary", "evaluate"])
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    a = ap.parse_args()
    if a.command == "train":
        train_models(a.models)
        write_training_summary()
    elif a.command == "summary":
        write_training_summary()
    else:
        evaluate()
