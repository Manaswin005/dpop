"""Phase 6 — Prediction service: one project in, prediction + risk + recommendations out.

Reuses the saved EBM, its feature list and class order (artifacts/model_metadata.json),
the SHAP code from Phase 5, and the Phase 6 risk and recommendation rules.

Run a demo:  python -m src.prediction.predictor
"""
from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd

from src.decision_support.recommendations import build_recommendations
from src.decision_support.risk import assign_risk
from src.explainability.shap_explain import explain
from src.utils.config import load_config, resolve

COST, FY, FLAG = "Lending Project Cost", "Approval FY", "lending_cost_reported"


class InputError(ValueError):
    """Raised when inputs cannot be used; `.messages` lists every problem."""

    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


class Predictor:
    def __init__(self):
        cfg = load_config()
        art = resolve(cfg["paths"]["artifacts_dir"])
        meta_path = art / "model_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("Model metadata missing. Run Phases 3–4 first (see README).")
        self.meta = json.loads(meta_path.read_text())
        model_path = art / "models" / self.meta["selected_model_file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Selected model file missing: {model_path}")
        self.model = joblib.load(model_path)
        self.order = self.meta["class_order"]
        self.cats = self.meta["features"]["categorical"]
        self.nums = self.meta["features"]["numerical"]
        self.columns = self.cats + self.nums

        # Reference values from the TRAINING set (allowed categories and value ranges)
        train = pd.read_csv(resolve(cfg["paths"]["processed_dir"]) / "train.csv")
        self.levels = {c: sorted(train[c].dropna().unique()) for c in self.cats}
        self.fy_range = (int(train[FY].min()), int(train[FY].max()))
        cost = train[COST].dropna()
        self.cost_range = (float(cost.min()), float(cost.max()))
        self.valid_fy = tuple(cfg["valid_approval_fy"])
        bg_path = art / "shap" / "background.csv"
        self.background = pd.read_csv(bg_path) if bg_path.exists() else None

    # ------------------------------------------------------------------
    def validate(self, inputs: dict) -> tuple[pd.DataFrame, list[str]]:
        """Check one project's inputs. Returns a 1-row frame and warnings, or raises InputError."""
        errors, warnings, row = [], [], {}

        for c in self.cats:
            v = inputs.get(c)
            if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
                errors.append(f"'{c}' is required.")
                continue
            v = str(v).strip()
            if v not in self.levels[c]:
                warnings.append(f"'{v}' was not seen in training data for {c}; the model gives it no effect.")
            row[c] = v

        fy = inputs.get(FY)
        try:
            fy = int(float(fy))
            if not self.valid_fy[0] <= fy <= self.valid_fy[1]:
                errors.append(f"Approval FY must be between {self.valid_fy[0]} and {self.valid_fy[1]}.")
            elif not self.fy_range[0] <= fy <= self.fy_range[1]:
                warnings.append(f"Approval FY {fy} is outside the training range {self.fy_range}; treat with extra caution.")
            row[FY] = fy
        except (TypeError, ValueError):
            errors.append("Approval FY is required and must be a year, e.g. 2010.")

        cost = inputs.get(COST)
        if cost is None or (isinstance(cost, float) and math.isnan(cost)) or str(cost).strip() == "":
            row[COST], row[FLAG] = np.nan, 0          # same convention as Phase 0: not reported
            warnings.append("No planned cost given; treated as 'not reported'.")
        else:
            try:
                cost = float(cost)
                if cost <= 0:
                    errors.append("Lending Project Cost must be a positive amount in USD.")
                elif not self.cost_range[0] <= cost <= self.cost_range[1]:
                    warnings.append(f"Cost ${cost:,.0f} is outside the training range; treat with extra caution.")
                row[COST], row[FLAG] = cost, 1
            except (TypeError, ValueError):
                errors.append("Lending Project Cost must be a number (USD) or left empty.")

        if errors:
            raise InputError(errors)
        return pd.DataFrame([row])[self.columns], warnings

    # ------------------------------------------------------------------
    def predict(self, inputs: dict, with_explanation: bool = True) -> dict:
        X, warnings = self.validate(inputs)
        proba = self.model.predict_proba(X)[0]
        risk = assign_risk(proba, self.order)

        contributions = None
        if with_explanation and self.background is not None:
            # Risk score = 1 - score(Successful), so its SHAP values are minus those of Successful
            sv = explain(self.model, self.background, X, self.cats, frames=[self.background, X])
            succ = self.order.index("Successful")
            contributions = {f: float(-v) for f, v in zip(X.columns, sv.values[0, :, succ])}
        elif with_explanation:
            warnings.append("SHAP background not found; run Phase 5 to enable explanations.")

        clean = X.iloc[0].to_dict()
        clean[COST] = None if pd.isna(clean[COST]) else clean[COST]
        return {
            "inputs": clean,
            "scores": {c: round(float(p), 3) for c, p in zip(self.order, proba)},
            **risk,
            "risk_contributions": ({k: round(v, 3) for k, v in sorted(contributions.items(), key=lambda kv: -abs(kv[1]))}
                                   if contributions else None),
            "recommendations": build_recommendations(risk["risk_level"], contributions, clean),
            "warnings": warnings,
            "model": self.meta["selected_model"],
        }


# Hypothetical example projects used in the README and the demo
EXAMPLES = {
    "Example L": {"Country Name": "China", "Sector Board": "Transport", "Agreement Type": "IBRD",
                  "Lending Instrument": "SIL", "Approval FY": 1995, "Lending Project Cost": 200_000_000},
    "Example A": {"Country Name": "India", "Sector Board": "Education", "Agreement Type": "IBRD",
                  "Lending Instrument": "SIL", "Approval FY": 2008, "Lending Project Cost": 150_000_000},
    "Example B": {"Country Name": "Kenya", "Sector Board": "Water", "Agreement Type": "IDA",
                  "Lending Instrument": "SIL", "Approval FY": 2010, "Lending Project Cost": 60_000_000},
    "Example C": {"Country Name": "Tanzania", "Sector Board": "Financial Sector", "Agreement Type": "IDA",
                  "Lending Instrument": "FIL", "Approval FY": 1985, "Lending Project Cost": None},
}


if __name__ == "__main__":
    p = Predictor()
    for name, inp in EXAMPLES.items():
        r = p.predict(inp)
        print(f"\n=== {name}: {inp}")
        print(f"Scores: {r['scores']}  ->  {r['predicted_outcome']} | Risk: {r['risk_level']} (score {r['risk_score']})")
        print(f"Why: {r['reason']}")
        for n in r["recommendations"]["feature_notes"]:
            print(f"  [{n['feature']} +{n['contribution']}] {n['note']}")
        for a in r["recommendations"]["actions"]:
            print(f"  - {a}")
        for w in r["warnings"]:
            print(f"  ! {w}")
