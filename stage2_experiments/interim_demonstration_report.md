# Phase 7 — Stage 2 Interim Risk Assessment Demonstration

## 1. Overview & Methodological Safeguards

This experiment provides an isolated, interpretable demonstration of how genuine, dated supervision signals from World Bank Implementation Status and Results Reports (ISRs) can be paired with the Stage 1 approval-time baseline.

> [!IMPORTANT]
> **Methodological Disclaimers:**
> - **No Machine-Learning Model Fitted:** With only 14 projects and 2 `Unsuccessful` instances, statistical classifiers cannot reliably estimate failure probabilities. No ML model was fitted.
> - **Documented Signals vs. Illustrative Heuristics:**
>   - *Documented Signals:* Task team ratings (`pdo_rating`, `ip_rating`), ledger drawdowns (`disbursement_pct`), and extension dates (`schedule_delay_months`) are authentic, timestamped historical records from disclosable ISRs.
>   - *Interim Risk Stance:* The assigned risk categories (`High Implementation Alert`, `Moderate Implementation Attention`, `Standard Supervision`) are **illustrative, transparent, and configurable decision-support rules**—NOT statistically proven failure predictions.
> - **No Dashboard Integration / Stage 1 Isolation:** Stage 1 code, models, artifacts, and dashboard pages remain 100% untouched.

---

## 2. Demonstration Results Table ($N = 14$)

Output exported to [`stage2_experiments/stage2_interim_assessment_demo.csv`](file:///c:/Users/Manas/OneDrive/Desktop/dpop_release/development-program-outcome-prediction/stage2_experiments/stage2_interim_assessment_demo.csv):

| Project ID | Stage 1 Baseline Risk | Interim Snapshot Date & Seq | Interim Documented Signals | Illustrative Interim Stance | Eventual Outcome (Target Only) |
|---|---|---|---|---|---|
| `P100968` | Medium | 2011-06-27 (Seq NaN) | *Ratings & disbursement unstated in disclosure* | Standard Supervision | Partially Successful |
| `P101614` | Medium | 2011-05-28 (Seq 2) | PDO downgraded to Mod. Unsatisfactory; Disb 0.0% | **High Implementation Alert** | Partially Successful |
| `P106752` | Medium | 2011-01-02 (Seq 5) | PDO Mod. Satisfactory; IP Mod. Satisfactory; Disb 10.0% | Moderate Attention | Partially Successful |
| `P108440` | Medium | 2011-03-28 (Seq 3) | PDO Satisfactory; IP Satisfactory; Disb low (1.0%) | Moderate Attention | Partially Successful |
| `P118560` | Medium | 2011-02-13 (Seq 2) | PDO Satisfactory; IP Satisfactory; Disb 31.0% | Standard Supervision | Partially Successful |
| `P121328` | Medium | 2012-01-08 (Seq 2) | PDO Satisfactory; IP Satisfactory; Disb 49.0% | Standard Supervision | Partially Successful |
| `P126033` | Medium | 2013-12-17 (Seq 2) | PDO Satisfactory; IP Mod. Satisfactory; Disb 76.0% | Moderate Attention | Partially Successful |
| `P126364` | Medium | 2012-10-22 (Seq 2) | PDO Satisfactory; IP Mod. Satisfactory; Disb low (0.0%) | Moderate Attention | Partially Successful |
| `P132101` | Medium | 2013-11-05 (Seq 2) | PDO Satisfactory; IP Satisfactory; Disb 19.0% | Standard Supervision | Partially Successful |
| `P106589` | Medium | 2011-02-26 (Seq 6) | PDO Mod. Satisfactory; IP Mod. Satisfactory; Disb low (4.0%) | Moderate Attention | Successful |
| `P113184` | Medium | 2012-12-14 (Seq 2) | PDO Mod. Satisfactory; IP Satisfactory; Disb 8.0% | Moderate Attention | Successful |
| `P118540` | Medium | 2011-10-05 (Seq 2) | IP Mod. Satisfactory; Disb low (2.0%) | Moderate Attention | Successful |
| `P126576` | Medium | 2012-12-15 (Seq 2) | PDO Mod. Satisfactory; IP Mod. Satisfactory; Disb low (0.0%) | Moderate Attention | **Unsuccessful** |
| `P126791` | Medium | 2014-06-21 (Seq 2) | PDO Mod. Satisfactory; IP Mod. Satisfactory; Disb 22.0% | Moderate Attention | **Unsuccessful** |

---

## 3. Key Analytical Insights

1. **Information Value of Interim Downgrades:**
   - In `P101614`, the Stage 1 baseline assigned a generic Medium risk at approval. The Sequence 2 ISR explicitly flagged a PDO rating downgrade to `Moderately Unsatisfactory` with `0.0%` disbursement, correctly shifting the project into `High Implementation Alert`.
2. **Disbursement Lag Detection:**
   - In `P126576` (an eventual `Unsuccessful` project), task-team ratings remained superficially `Moderately Satisfactory` at Sequence 2, but actual disbursement was completely stalled at `0.0%` one year into execution.
3. **Preservation of Scientific Integrity:**
   - Because $N = 14$, these rules illustrate how a decision-support dashboard could surface operational alerts for portfolio managers. They do not claim validated causal prediction.
