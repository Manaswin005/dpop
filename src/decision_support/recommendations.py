"""Phase 6 — Rule-based recommendations (architecture Stage 10).

Two simple layers:
  1. General actions for the risk level (templates below).
  2. Feature notes for the features that raised this project's risk score the
     most, according to SHAP. Wording describes association with the model's
     score; it never claims a feature causes failure or that changing it helps.
"""
from __future__ import annotations

# Raise a feature note only if it moved the risk score up by at least this much
MIN_CONTRIBUTION = 0.02
MAX_FEATURE_NOTES = 3

LEVEL_ACTIONS = {
    "Low": [
        "Proceed with standard appraisal and supervision arrangements.",
        "Keep routine monitoring of results indicators.",
    ],
    "Medium": [
        "Review implementation readiness (agency capacity, procurement and staffing plans) before approval.",
        "Agree a results framework with clear mid-term review triggers.",
        "Schedule more frequent supervision during the first two years.",
    ],
    "High": [
        "Refer the project for a detailed readiness and design review before approval.",
        "Compare the design with completed projects of the same type, sector and country, and note lessons learned.",
        "Set up early-warning indicators and a planned mid-term restructuring checkpoint.",
        "Confirm that risk-mitigation measures and contingency resources are documented.",
    ],
}

FEATURE_NOTES = {
    "Country Name": "Projects in {value} were associated with a higher model risk score. Review country-level "
                    "implementation capacity and results of earlier projects in {value}.",
    "Sector Board": "The {value} sector was associated with a higher model risk score. Draw on sector "
                    "specialists and on evaluations of comparable sector projects.",
    "Lending Instrument": "The {value} lending instrument was associated with a higher model risk score. "
                          "Check that the instrument suits the project's objectives and the borrower's capacity.",
    "Agreement Type": "The {value} financing type was associated with a higher model risk score. Review "
                      "financing arrangements and fiduciary requirements.",
    "Approval FY": "The approval year ({value}) raised the model risk score. This mostly reflects historical "
                   "rating patterns, so treat it as context rather than something to act on.",
    "Lending Project Cost": "The planned cost ({value}) was associated with a higher model risk score. "
                            "Check that the cost estimate and implementation schedule are realistic.",
    "lending_cost_reported": "No planned cost was recorded. Record a cost estimate so the project can be "
                             "appraised and monitored properly.",
}

HUMAN_REVIEW = ("These are decision-support prompts based on patterns in historical World Bank projects. "
                "They are not causal findings, and final decisions need expert human review.")


def _fmt(feature: str, value) -> str:
    if feature == "Lending Project Cost":
        return "not reported" if value is None else f"${float(value) / 1e6:,.1f}M"
    if feature == "Approval FY":
        return str(int(value))
    return str(value)


def _note_text(feature: str, inputs: dict) -> str | None:
    """Pick the note that matches the actual input value (or None if no note applies)."""
    cost_missing = inputs.get("Lending Project Cost") is None
    if feature in ("Lending Project Cost", "lending_cost_reported"):
        if cost_missing:
            return FEATURE_NOTES["lending_cost_reported"]       # "No planned cost was recorded..."
        if feature == "lending_cost_reported":
            return None   # cost WAS reported; the flag mainly tracks approval era, so no action note
    if feature not in FEATURE_NOTES:
        return None
    return FEATURE_NOTES[feature].format(value=_fmt(feature, inputs.get(feature)))


def build_recommendations(risk_level: str, contributions: dict | None, inputs: dict) -> dict:
    """Combine level actions with feature notes.

    contributions: {feature: change in risk score} from SHAP (positive = raised risk).
    Pass None when explanations are unavailable; level actions are still returned.
    """
    if risk_level not in LEVEL_ACTIONS:
        raise ValueError(f"Unknown risk level: {risk_level}")
    notes = []
    if contributions:
        raising = sorted(((f, c) for f, c in contributions.items() if c >= MIN_CONTRIBUTION),
                         key=lambda fc: fc[1], reverse=True)[:MAX_FEATURE_NOTES]
        for feature, contrib in raising:
            text = _note_text(feature, inputs)
            # Skip notes that do not apply, and never repeat the same note twice
            if text and text not in [n["note"] for n in notes]:
                notes.append({"feature": feature, "contribution": round(float(contrib), 3), "note": text})
    return {"actions": LEVEL_ACTIONS[risk_level], "feature_notes": notes, "disclaimer": HUMAN_REVIEW}
