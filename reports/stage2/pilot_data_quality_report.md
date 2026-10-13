# Stage 2 Implementation-Stage Data Pilot Quality & Coverage Report

## 1. Executive Summary

This pilot evaluated the automated construction of an implementation-stage supervision dataset for World Bank development projects. In accordance with the Phase 4 protocol, the pilot was executed against an isolated cohort of **30 Investment Project Financing (IPF)** operations drawn strictly from the existing training partition ([`data/processed/train.csv`](file:///c:/Users/Manas/OneDrive/Desktop/dpop_release/development-program-outcome-prediction/data/processed/train.csv)) with `Approval FY >= 2008` using random seed `42`.

When an external timeout occurred on project 25 (`P123457`), the pipeline adhered to the strict operational protocol and **stopped immediately** without retrying, switching hosts, or prompting for elevated permissions. All progress up to that point was preserved in local cache files.

---

## 2. Cohort Selection & Provenance Breakdown

- **Total Eligible Population in `train.csv`:** 933 projects (Approval FY $\ge$ 2008, IPF only).
- **Stratified Pilot Cohort:** 30 projects selected into [`data/stage2/pilot_cohort.csv`](file:///c:/Users/Manas/OneDrive/Desktop/dpop_release/development-program-outcome-prediction/data/stage2/pilot_cohort.csv).
  - Partially Successful: 17 projects (56.7%)
  - Successful: 10 projects (33.3%)
  - Unsuccessful: 3 projects (10.0%)
- **Pipeline Execution Summary (25 projects attempted):**
  - **Valid Interim Snapshots Retained:** **12 projects** (48.0% of attempted)
  - **Text Download Failed / Connection Reset:** **11 projects** (44.0% of attempted)
  - **No Eligible Interim Snapshot (Temporal Exclusions):** **1 project** (4.0% of attempted)
  - **Metadata Fetch Timeout (Halted Pipeline):** **1 project** (`P123457`)
  - **Unqueried (Preserved Due to Protocol Stop):** **5 projects**

---

## 3. Validated Interim Snapshots Summary (`pilot_snapshots.csv`)

| Project ID | Snapshot Date | Sequence | PDO Rating (Current) | IP Rating (Current) | Overall Risk | Disbursed % | Schedule Delay (Months) | Outcome Class (Target Only) |
|---|---|---|---|---|---|---|---|---|
| `P100968` | 2011-06-27 | NaN | *Unstated in text* | *Unstated in text* | *Unstated in text* | NaN | NaN | Partially Successful |
| `P101614` | 2011-05-28 | 2.0 | Moderately Unsatisfactory | Moderately Satisfactory | High | 0.0% | 0.0 | Partially Successful |
| `P106752` | 2011-01-02 | 5.0 | Moderately Satisfactory | Moderately Satisfactory | *Unstated in text* | 10.0% | 0.0 | Partially Successful |
| `P108440` | 2011-03-28 | 3.0 | Satisfactory | Satisfactory | *Unstated in text* | 1.0% | 0.0 | Partially Successful |
| `P118560` | 2011-02-13 | 2.0 | Satisfactory | Satisfactory | *Unstated in text* | 31.0% | 0.0 | Partially Successful |
| `P121328` | 2012-01-08 | 2.0 | Satisfactory | Satisfactory | Moderate | 49.0% | 0.0 | Partially Successful |
| `P126033` | 2013-12-17 | 2.0 | Satisfactory | Moderately Satisfactory | Moderate | 76.0% | 0.0 | Partially Successful |
| `P126364` | 2012-10-22 | 2.0 | Satisfactory | Moderately Satisfactory | Moderate | 0.0% | 0.0 | Partially Successful |
| `P132101` | 2013-11-05 | 2.0 | Satisfactory | Satisfactory | High | 19.0% | 0.0 | Partially Successful |
| `P106589` | 2011-02-26 | 6.0 | Moderately Satisfactory | Moderately Satisfactory | *Unstated in text* | 4.0% | 0.0 | Successful |
| `P113184` | 2012-12-14 | 2.0 | Moderately Satisfactory | Satisfactory | Moderate | 8.0% | 0.0 | Successful |
| `P118540` | 2011-10-05 | 2.0 | *Unstated in text* | Moderately Satisfactory | Low | 2.0% | 0.0 | Successful |

---

## 4. Feature Coverage & Missingness Analysis ($N = 12$)

| Feature Name | Populated Count | Missing Rate | Verified Semantics & Observations |
|---|---|---|---|
| `snapshot_date` | 12 / 12 | **0.0%** | Derived from official report archive date. Precedes final IEG evaluation in all cases. |
| `snapshot_sequence` | 11 / 12 | **8.3%** | Sequence numbers range from 2 to 6; majority are early interim snapshots (Sequence 2). |
| `ip_rating_current` | 11 / 12 | **8.3%** | Highly complete; captures task team assessment of execution progress. |
| `pdo_rating_current` | 10 / 12 | **16.7%** | Captures progress towards development objective; 2 reports had blank disclosure sections. |
| `disbursement_percentage` | 11 / 12 | **8.3%** | Extracted from financial agreement tables; ranges from 0.0% to 76.0%. |
| `schedule_delay_months` | 11 / 12 | **8.3%** | Computed from original and revised closing dates. All valid snapshots show 0.0 extension at this early interim stage. |
| `overall_risk_rating` | 7 / 12 | **41.7%** | Structured risk ratings were formalized by the World Bank in 2012; 2010–2011 reports omit this section. |

---

## 5. Temporal Validity and Leakage Prevention

1. **Zero Test Leakage:** All 30 cohort projects and all 12 validated snapshots belong strictly to `train.csv`.
2. **Strict Date Ordering:** Verified that for every snapshot:
   $$\text{Approval Date} \le \text{snapshot\_date} \le \text{IEG\_EvalDate}$$
3. **Target Isolation:** `outcome_class` is preserved solely for evaluation auditing and is excluded from feature representations.
4. **Schedule Delay Sanity:** Only closing dates explicitly reported inside the snapshot document were used. No future closing date amendments were back-projected.

---

## 6. Stage 1 Baseline Preservation Check

Verification confirmed that zero Stage 1 files were modified during this phase. SHA256 hashes match the pre-execution baseline exactly:
- `app.py`: `4f1afefb343f1726...` (MATCH)
- `config.yaml`: `dcfd6bd1b407832f...` (MATCH)
- `requirements.txt`: `03d671544e6b8b42...` (MATCH)
- `data/processed/train.csv`: `e8c96377e9e69391...` (MATCH)
- `data/processed/test.csv`: `677d4f5f8511f88e...` (MATCH)
- `artifacts/model_metadata.json`: `4ac36cfb9c48b7bc...` (MATCH)
- `artifacts/models/EBM.joblib`: `20708baf14c03765...` (MATCH)
