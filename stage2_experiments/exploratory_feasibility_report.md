# Stage 2 Exploratory Modeling Feasibility Report

## 1. Class Counts & Sample Representation

The verified Stage 2 pilot dataset ([`data/stage2/pilot_snapshots.csv`](file:///c:/Users/Manas/OneDrive/Desktop/dpop_release/development-program-outcome-prediction/data/stage2/pilot_snapshots.csv)) contains **14 validated project snapshots**:

| Final Outcome Class | Snapshot Count | Percentage |
|---|---|---|
| **Partially Successful** | 9 | 64.3% |
| **Successful** | 3 | 21.4% |
| **Unsuccessful** | **2** | **14.3%** |
| **Total** | **14** | **100.0%** |

### Data Integrity & Verification Audit
- **Project Overlap:** 14 unique projects. 100% (14/14) belong strictly to the Stage 1 training cohort (`data/processed/train.csv`). 0% (0/14) appear in `data/processed/test.csv`.
- **Chronological Ordering & Temporal Validity:** Every snapshot strictly satisfies $\text{Approval Date} < \text{snapshot\_date} < \text{IEG\_EvalDate}$.
- **Target Leakage Safeguard:** Live predictors exclude `outcome_class`, evaluation scores, and provenance identifiers. Only genuine interim metrics available on or before the snapshot date are used.
- **Missing Value Profile:**
  - `pdo_rating_current`: 2 missing (14.3%)
  - `ip_rating_current`: 1 missing (7.1%)
  - `overall_risk_rating`: 5 missing (35.7% — not used as predictor)
  - `disbursement_percentage`: 1 missing (7.1%)
  - `schedule_delay_months`: 1 missing (7.1%), 13 instances confirmed $0.0$ delay.

---

## 2. Model Viability Assessment

> [!CAUTION]
> **Determination: Supervised machine-learning model fitting (binary or three-class) is NOT statistically or methodologically justified on this 14-record dataset.**
> Attempting to train or cross-validate a predictive model (e.g., Logistic Regression, Random Forest, or EBM) would be scientifically invalid for the following reasons:
>
> 1. **Extreme Minority Class Scarcity ($N_{\text{Unsuccessful}} = 2$, $N_{\text{Successful}} = 3$):**
>    - In any project-level cross-validation split (e.g., 5-fold CV), 3 out of 5 folds would contain **zero** instances of `Unsuccessful`, making precision, recall, and ROC-AUC degenerate or mathematically undefined.
> 2. **Severe Overfitting & Parameter Variance:**
>    - Fitting 4–6 feature weights on only 14 observations produces extreme parameter instability and cannot generalize.
> 3. **Non-Uniform Temporal Timing:**
>    - Snapshots range from Sequence 2 (early implementation) to Sequence 6, with elapsed calendar time ranging from 6 months to 2.5 years post-approval. Without a standardized observation horizon, estimating risk weights would confound implementation progress with calendar age.

---

## 3. Descriptive Alignment with Stage 1 Baseline

Rather than fitting a degenerate classifier, the 14 implementation-stage snapshots were compared in a strictly isolated analysis against Stage 1 approval-time predictions ([`stage2_experiments/pilot_comparison_table.csv`](file:///c:/Users/Manas/OneDrive/Desktop/dpop_release/development-program-outcome-prediction/stage2_experiments/pilot_comparison_table.csv)):

| Project ID | Actual Outcome | Stage 1 EBM Pred | Stage 1 Probs (S / P / U) | Snapshot Date | Seq | PDO Rating | IP Rating | Disb % | Schedule Delay |
|---|---|---|---|---|---|---|---|---|---|
| `P100968` | Partially Successful | Successful | 0.49 / 0.45 / 0.06 | 2011-06-27 | *None* | *None* | *None* | *None* | *None* |
| `P101614` | Partially Successful | Partially Successful | 0.15 / 0.66 / 0.18 | 2011-05-28 | 2 | Mod. Unsatisfactory | Mod. Satisfactory | 0.0% | 0.0 mo |
| `P106752` | Partially Successful | Partially Successful | 0.20 / 0.58 / 0.22 | 2011-01-02 | 5 | Mod. Satisfactory | Mod. Satisfactory | 10.0% | 0.0 mo |
| `P108440` | Partially Successful | Partially Successful | 0.13 / 0.62 / 0.25 | 2011-03-28 | 3 | Satisfactory | Satisfactory | 1.0% | 0.0 mo |
| `P118560` | Partially Successful | Partially Successful | 0.15 / 0.54 / 0.30 | 2011-02-13 | 2 | Satisfactory | Satisfactory | 31.0% | 0.0 mo |
| `P121328` | Partially Successful | Partially Successful | 0.18 / 0.62 / 0.20 | 2012-01-08 | 2 | Satisfactory | Satisfactory | 49.0% | 0.0 mo |
| `P126033` | Partially Successful | Partially Successful | 0.30 / 0.59 / 0.11 | 2013-12-17 | 2 | Satisfactory | Mod. Satisfactory | 76.0% | 0.0 mo |
| `P126364` | Partially Successful | Partially Successful | 0.36 / 0.54 / 0.10 | 2012-10-22 | 2 | Satisfactory | Mod. Satisfactory | 0.0% | 0.0 mo |
| `P132101` | Partially Successful | Partially Successful | 0.27 / 0.57 / 0.17 | 2013-11-05 | 2 | Satisfactory | Satisfactory | 19.0% | 0.0 mo |
| `P106589` | Successful | Partially Successful | 0.31 / 0.62 / 0.07 | 2011-02-26 | 6 | Mod. Satisfactory | Mod. Satisfactory | 4.0% | 0.0 mo |
| `P113184` | Successful | Partially Successful | 0.33 / 0.58 / 0.08 | 2012-12-14 | 2 | Mod. Satisfactory | Satisfactory | 8.0% | 0.0 mo |
| `P118540` | Successful | Partially Successful | 0.21 / 0.60 / 0.18 | 2011-10-05 | 2 | *None* | Mod. Satisfactory | 2.0% | 0.0 mo |
| `P126576` | **Unsuccessful** | Partially Successful | 0.20 / 0.54 / 0.26 | 2012-12-15 | 2 | Mod. Satisfactory | Mod. Satisfactory | **0.0%** | 0.0 mo |
| `P126791` | **Unsuccessful** | Partially Successful | 0.31 / 0.57 / 0.12 | 2014-06-21 | 2 | Mod. Satisfactory | Mod. Satisfactory | **22.0%** | 0.0 mo |

### Key Analytical Findings
1. **Approval-Stage Base Rate:** Stage 1 predicted `Partially Successful` for 13 of 14 projects, consistent with the empirical base rate for post-2008 IPF operations.
2. **Early Implementation Downgrade (`P101614`):** Task team recorded a PDO downgrade to `Moderately Unsatisfactory` with `0.0%` disbursement at Sequence 2, providing a clear operational distress signal that was unavailable at approval.
3. **Severe Disbursement Lag in Failed Projects (`P126576`):** The project eventually failed (`Unsuccessful`), and while supervision ratings were still reported as `Moderately Satisfactory` at Sequence 2, cumulative disbursement had reached only `0.0%` seven months into implementation.
4. **Supervision Rating Inertia:** In both `Unsuccessful` projects (`P126576` and `P126791`), task teams maintained `Moderately Satisfactory` ratings during early supervision. This indicates that task-team ratings alone lag real execution troubles, reinforcing the need for quantitative indicators like disbursement trajectory.

---

## 4. Minimum Conditions & Recommendation

### Minimum Conditions for Stage 2 Supervised Modeling
To justify training a machine-learning model or evaluating formal performance metrics:
1. **Sample Size:** $N \ge 100$ projects with authentic dated ISR snapshots.
2. **Class Representation:** At least 20–30 genuine `Unsuccessful` instances to establish a statistically meaningful failure decision boundary.
3. **Temporal Normalization:** Uniform snapshot observation windows (e.g., standardizing at 18–24 months post-approval).

### Operational Recommendation
Maintain the **isolated, transparent rule-based heuristic stance** ([`stage2_experiments/interim_risk_assessment.py`](file:///c:/Users/Manas/OneDrive/Desktop/dpop_release/development-program-outcome-prediction/stage2_experiments/interim_risk_assessment.py)) as demonstrated in the prototype dashboard view. Do not train machine-learning models or make unsubstantiated claims of predictive accuracy.
