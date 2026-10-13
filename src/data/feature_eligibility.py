"""Feature-eligibility register: one documented decision for every column.

Decision values:
  INCLUDE      - approval-time predictive feature
  EXCLUDE      - not used as a predictor (leakage, identifier, redundant or unreliable)
  TARGET       - the label only
  AUDIT_ONLY   - kept for grouping / record selection / traceability, never a predictor

Audit evidence quoted here comes from src/data/audit.py output (Phase 0).
"""
from __future__ import annotations

import pandas as pd

ELIGIBILITY = [
    # ---- identifiers -------------------------------------------------------
    ("Project ID", "AUDIT_ONLY", "Identifier. Used for project-aware grouping and traceability.",
     "Assigned at concept stage; carries no outcome information by design."),
    ("Project Name", "EXCLUDE", "Free-text identifier; would let a model memorise specific projects.",
     "Known at approval, but not a legitimate generalisable predictor."),
    # ---- approval-time descriptors ----------------------------------------
    ("Region", "INCLUDE", "Geographic context. AFE/AFW harmonised to AFR (post-2020 sub-region codes applied unevenly by era).",
     "Known at approval. Audit: AFE/AFW appear mainly on post-1990 projects, AFR on older ones in the same countries."),
    ("Country Code", "EXCLUDE", "Corrupted upstream: Namibia's ISO code 'NA' was read as missing and filled with 'IN' (India).",
     "Known at approval, but unreliable in this file. Country Name carries the same information correctly."),
    ("Country Name", "INCLUDE", "Borrower country context (182 countries; rare levels grouped later on training data).",
     "Known at approval."),
    ("Approval Date", "AUDIT_ONLY", "Used only to repair one invalid Approval FY value.",
     "Known at approval. Approval FY carries the same timing information."),
    ("Approval FY", "INCLUDE", "Approval era: lending practices and evaluation standards changed over time.",
     "Known at approval. One invalid value (0) repaired from Approval Date."),
    ("Sector Board", "INCLUDE", "Sector of the operation (rare categories grouped later on training data).",
     "Assigned at approval; taxonomy changed over eras, which is noted as a limitation."),
    ("Agreement Type", "INCLUDE", "Financing source (IBRD, IDA, trust funds).",
     "Known at approval. Overlaps strongly with Product Line Code; redundancy assessed in feature selection."),
    ("Lending Project Cost", "INCLUDE", "Planned total project cost at approval (log scale later).",
     "Planned figure. Audit: 42.7% of rows hold one identical value (55,700,000) = upstream fill; "
     "converted to missing + indicator `lending_cost_reported`. Nearly all pre-1990 projects are affected."),
    ("Net Commitment", "EXCLUDE", "Timing NOT established; evidence indicates post-approval adjustment.",
     "Audit: 269 negative values, 76% below project cost, 25.5% non-round amounts. "
     "Consistent with net-of-cancellation figures recorded after implementation."),
    ("Lending Instrument Type", "INCLUDE", "IPF / DPF / PforR.", "Known at approval."),
    ("Lending Instrument", "INCLUDE", "Detailed instrument (SIL, DPL, TAL ...).", "Known at approval."),
    ("Product Line Code", "INCLUDE", "Product line (IBRD/IDA, GEF, ...).", "Known at approval."),
    ("Product Line", "EXCLUDE", "One-to-one duplicate of Product Line Code.", "Known at approval; redundant."),
    # ---- post-approval / completion ---------------------------------------
    ("Deactivation Date", "EXCLUDE", "Project closing date.", "Post-completion."),
    ("Exit FY", "EXCLUDE", "Fiscal year of exit; implies duration.", "Post-completion."),
    ("ERR at Appraisal", "EXCLUDE",
     "Unreliable in this file: 65.8% of rows hold one identical value (21.0) = upstream fill, "
     "and values differ between evaluations of the same project.",
     "Concept is appraisal-time, but values here are transcribed from later IEG evaluation records; "
     "genuine values cannot be separated from filled ones. Excluded from the primary model."),
    ("ERR at Completion", "EXCLUDE", "Economic return measured at completion.", "Post-completion (69.9% upstream-filled with 17.0)."),
    # ---- IEG evaluation fields -------------------------------------------
    ("IEG_EvalDate", "AUDIT_ONLY", "Selects the latest evaluation per project.", "Post-completion evaluation."),
    ("IEG_EvalFY", "EXCLUDE", "Evaluation year.", "Post-completion evaluation."),
    ("IEG_EvalType", "AUDIT_ONLY", "Tie-breaker for record selection.", "Post-completion evaluation."),
    ("IEG_Outcome", "TARGET", "Mapped to three outcome classes.", "Post-completion evaluation."),
]
_POST_RATINGS = [
    "(disc)IEG_IDImpact", "IEG_BankQualityAtEntry", "IEG_BankQualityOfSupervision",
    "IEG_OverallBankPerf", "(disc)IEG_BorrPrep", "IEG_ImplementingAgencyPerf",
    "IEG_GovernmentPerf", "IEG_OverallBorrPerf", "IEG_ICRQuality",
    "(disc)IEG_Sustainability", "IEG_MEQuality",
]
for _c in _POST_RATINGS:
    _note = ("Rated retrospectively by IEG even though it describes preparation quality."
             if _c in ("IEG_BankQualityAtEntry", "(disc)IEG_BorrPrep") else "IEG performance rating.")
    ELIGIBILITY.append((_c, "EXCLUDE", _note, "Post-completion evaluation."))
ELIGIBILITY.append(("IEG_SourceDocumentURL", "EXCLUDE", "Link to the evaluation document.", "Post-completion evaluation."))


def eligibility_table() -> pd.DataFrame:
    return pd.DataFrame(ELIGIBILITY, columns=["column", "decision", "rationale", "timing_evidence"])


def included_features() -> list[str]:
    t = eligibility_table()
    return t.loc[t["decision"] == "INCLUDE", "column"].tolist()
