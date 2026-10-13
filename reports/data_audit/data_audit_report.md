# Phase 0 — Dataset Audit Report

Source file: `data/raw/cleaned_development_program_dataset.csv` — 12,932 rows × 35 columns. Missing cells as supplied: 0.

## 1. Target mapping
Project-specific three-class grouping of IEG outcome ratings (not an official IEG convention):

| IEG rating | Class |
|---|---|
| Highly Satisfactory | Successful |
| Satisfactory | Successful |
| Moderately Satisfactory | Partially Successful |
| Moderately Unsatisfactory | Partially Successful |
| Unsatisfactory | Unsuccessful |
| Highly Unsatisfactory | Unsuccessful |

Excluded labels: {'Not Rated': 162, 'Not Available': 9, 'Not Applicable': 9} → 180 rows removed; 12,752 rows kept.

## 2. Repeated Project IDs
- Exact duplicate rows: 0.
- 2,026 Project IDs appear more than once (4,059 rows); group sizes {'2': 2019, '3': 7}.
- Most common patterns: {'ES|PAR': 1019, 'PAR|PCR': 750, 'EVM|PAR': 219, 'ES|ES': 28, 'ES|ES|PAR': 3, 'ES|ES|ES': 3}.
- Only evaluation fields differ within these groups; approval-time fields are identical. These are **separate IEG evaluations of the same project**, not separate projects.
- 775 groups carry different raw outcome ratings.

**Decision:** keep one rated evaluation per project — the most recent (latest IEG judgement), tie-broken by evaluation-type priority (PAR first). Rows removed: 2,002. Sensitivity: choosing the earliest evaluation instead would change the three-class label for 564 of 1,995 multi-evaluation projects. Train/test splitting will still be grouped by Project ID as a safeguard.

## 3. Upstream fill values (data-quality finding)
The supplied file reports zero missing values, but several numeric columns hold one constant in a large share of rows:

| Column | Repeated value | Share of rows |
|---|---|---|
| Lending Project Cost | 55700000.0 | 42.7% |
| ERR at Appraisal | 21.0 | 65.8% |
| ERR at Completion | 17.0 | 69.9% |

These are median fills applied before the file was supplied. Treating them as real values would mislead the models.

## 4. Key eligibility decisions
**Net Commitment — EXCLUDED.** 269 negative values, 76% below project cost, 25.6% non-round amounts (vs 2.7% for project cost). This pattern indicates amounts net of later cancellations; approval-time availability cannot be established.

**ERR at Appraisal — EXCLUDED from the primary model.** 65.8% of rows equal 21.0 (next most common value: 1.5%), so the true missingness is hidden. Values also differ between evaluations of the same project in 391 projects, showing they were transcribed from evaluation records. Genuine appraisal values cannot be separated from filled ones.

**Lending Project Cost — INCLUDED with repair.** The fill value is converted back to missing and an indicator `lending_cost_reported` is added; 57.9% of modelling rows have a reported cost. Imputation will be fitted on training data only.

**Approval FY:** 1 invalid value repaired from Approval Date; 24 minor date/FY mismatches left as recorded.

**Region:** AFE/AFW harmonised to AFR. Counts: {'AFR': 3199, 'LCR': 2123, 'EAP': 1734, 'ECA': 1473, 'SAR': 1264, 'MNA': 951, 'OTH': 6}.

**Redundant pairs:** {'Country Code <-> Country Name one-to-one': False, 'Country codes shared by several countries': {'IN': ['India', 'Namibia']}, 'Product Line Code <-> Product Line one-to-one': True}

## 5. Modelling dataset after Phase 0 rules
10,750 projects, approval FY 1948–2019.

| Class | Count | Share |
|---|---|---|
| Successful | 5,316 | 49.5% |
| Partially Successful | 3,446 | 32.1% |
| Unsuccessful | 1,988 | 18.5% |

Included approval-time features: Region, Country Name, Approval FY, Sector Board, Agreement Type, Lending Project Cost, Lending Instrument Type, Lending Instrument, Product Line Code (+ derived `lending_cost_reported`).

## 6. Feature-eligibility register

| Column | Decision | Rationale | Timing evidence |
|---|---|---|---|
| Project ID | AUDIT_ONLY | Identifier. Used for project-aware grouping and traceability. | Assigned at concept stage; carries no outcome information by design. |
| Project Name | EXCLUDE | Free-text identifier; would let a model memorise specific projects. | Known at approval, but not a legitimate generalisable predictor. |
| Region | INCLUDE | Geographic context. AFE/AFW harmonised to AFR (post-2020 sub-region codes applied unevenly by era). | Known at approval. Audit: AFE/AFW appear mainly on post-1990 projects, AFR on older ones in the same countries. |
| Country Code | EXCLUDE | Corrupted upstream: Namibia's ISO code 'NA' was read as missing and filled with 'IN' (India). | Known at approval, but unreliable in this file. Country Name carries the same information correctly. |
| Country Name | INCLUDE | Borrower country context (182 countries; rare levels grouped later on training data). | Known at approval. |
| Approval Date | AUDIT_ONLY | Used only to repair one invalid Approval FY value. | Known at approval. Approval FY carries the same timing information. |
| Approval FY | INCLUDE | Approval era: lending practices and evaluation standards changed over time. | Known at approval. One invalid value (0) repaired from Approval Date. |
| Sector Board | INCLUDE | Sector of the operation (rare categories grouped later on training data). | Assigned at approval; taxonomy changed over eras, which is noted as a limitation. |
| Agreement Type | INCLUDE | Financing source (IBRD, IDA, trust funds). | Known at approval. Overlaps strongly with Product Line Code; redundancy assessed in feature selection. |
| Lending Project Cost | INCLUDE | Planned total project cost at approval (log scale later). | Planned figure. Audit: 42.7% of rows hold one identical value (55,700,000) = upstream fill; converted to missing + indicator `lending_cost_reported`. Nearly all pre-1990 projects are affected. |
| Net Commitment | EXCLUDE | Timing NOT established; evidence indicates post-approval adjustment. | Audit: 269 negative values, 76% below project cost, 25.5% non-round amounts. Consistent with net-of-cancellation figures recorded after implementation. |
| Lending Instrument Type | INCLUDE | IPF / DPF / PforR. | Known at approval. |
| Lending Instrument | INCLUDE | Detailed instrument (SIL, DPL, TAL ...). | Known at approval. |
| Product Line Code | INCLUDE | Product line (IBRD/IDA, GEF, ...). | Known at approval. |
| Product Line | EXCLUDE | One-to-one duplicate of Product Line Code. | Known at approval; redundant. |
| Deactivation Date | EXCLUDE | Project closing date. | Post-completion. |
| Exit FY | EXCLUDE | Fiscal year of exit; implies duration. | Post-completion. |
| ERR at Appraisal | EXCLUDE | Unreliable in this file: 65.8% of rows hold one identical value (21.0) = upstream fill, and values differ between evaluations of the same project. | Concept is appraisal-time, but values here are transcribed from later IEG evaluation records; genuine values cannot be separated from filled ones. Excluded from the primary model. |
| ERR at Completion | EXCLUDE | Economic return measured at completion. | Post-completion (69.9% upstream-filled with 17.0). |
| IEG_EvalDate | AUDIT_ONLY | Selects the latest evaluation per project. | Post-completion evaluation. |
| IEG_EvalFY | EXCLUDE | Evaluation year. | Post-completion evaluation. |
| IEG_EvalType | AUDIT_ONLY | Tie-breaker for record selection. | Post-completion evaluation. |
| IEG_Outcome | TARGET | Mapped to three outcome classes. | Post-completion evaluation. |
| (disc)IEG_IDImpact | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| IEG_BankQualityAtEntry | EXCLUDE | Rated retrospectively by IEG even though it describes preparation quality. | Post-completion evaluation. |
| IEG_BankQualityOfSupervision | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| IEG_OverallBankPerf | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| (disc)IEG_BorrPrep | EXCLUDE | Rated retrospectively by IEG even though it describes preparation quality. | Post-completion evaluation. |
| IEG_ImplementingAgencyPerf | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| IEG_GovernmentPerf | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| IEG_OverallBorrPerf | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| IEG_ICRQuality | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| (disc)IEG_Sustainability | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| IEG_MEQuality | EXCLUDE | IEG performance rating. | Post-completion evaluation. |
| IEG_SourceDocumentURL | EXCLUDE | Link to the evaluation document. | Post-completion evaluation. |

_Environment: Python 3.11.17, pandas 2.2.3, Linux-6.18.44-fc-v114-x86_64-with-glibc2.39._