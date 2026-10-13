# Development Program Outcome Prediction System (DPOP)

**Using Advanced Ensemble Learning, Explainable AI and Risk-Based Insights**

DPOP predicts the likely outcome of a World Bank development project as **Successful**, **Partially Successful** or **Unsuccessful**. It uses only information available **at project approval**.

It explains each prediction with SHAP, assigns a Low, Medium or High risk level, suggests rule-based actions, and summarises risk across the whole portfolio for planning. It is a decision-support tool. It does not make causal claims and does not guarantee project success.

---

## 1. Quick start (Windows PowerShell)

Run these from inside the `development-program-outcome-prediction` folder.

```powershell
# 1. Create and activate a virtual environment (Python 3.12)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Install dependencies (downloads PyTorch, which is large; it is only needed for TabPFN)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the tests (about 1 minute; no training, no downloads)
python -m pytest -q

# 4. Launch the dashboard (opens http://localhost:8501; stop with Ctrl + C)
streamlit run app.py
```

**Python version note:**
- The project was developed and tested on **Python 3.11.17 (Linux)**.
- On Python 3.12 the pinned requirements resolved successfully in a dry run (Linux), but the tests and the app have **not yet been run on 3.12**.
- If anything fails on 3.12, use `py -3.11` instead.
- Nothing has been verified on Windows yet (see section 12).

---

## 2. Three ways to run the project

| Goal | Commands | Time | Changes saved files? |
|---|---|---|---|
| **A. Demo (normal use)** | `streamlit run app.py` | Seconds | No. The app only reads saved models and reports. |
| **B. Re-evaluate saved models** | `python -m src.models.train evaluate` | Under a minute | Rewrites the Phase 4 reports and charts, `model_metadata.json` and `Ensemble.joblib` (rebuilt from the saved member models). No retraining. |
| **C. Rebuild everything from the raw CSV** | Full order below | About 20–25 min on the 2-vCPU dev machine; longer on slower PCs | **Yes. It overwrites trained models and results.** Only do this on purpose. |

**Full rebuild order (option C).** Each step uses the previous step's outputs.

```powershell
python -m src.data.audit                    # Stage 0: dataset audit + feature eligibility
python -m src.preprocessing.prepare         # Stages 1, 4: cleaning + stratified 80:20 split
python -m src.eda.eda                       # Stage 2: EDA charts (training set only)
python -m src.eda.fy_cutoff_check           # FY1990 rating-scale check (descriptive)
python -m src.features.selection            # Stage 3: feature selection (training set only)
python -m src.models.train train            # Stage 5: 5-fold CV + final fit  (SLOW, ~20-25 min)
python -m src.models.train summary          # Phase 3 training table
python -m src.models.train evaluate         # Stages 6-7: select by CV, score test set once
python -m src.explainability.shap_explain   # Stage 8: SHAP (~35 s)
python -m src.decision_support.risk         # Stage 9: risk-rule check on training data
python -m src.prediction.predictor          # Stages 9-10: example predictions + recommendations
python -m src.decision_support.policy       # Stage 13: portfolio policy report
streamlit run app.py                        # Stages 11-12: dashboard
```

To retrain **one** model only, for example GPC: `python -m src.models.train train --models GPC`, then `python -m src.models.train evaluate`.

---

## 3. Project structure

```
development-program-outcome-prediction/
├── app.py                          # Streamlit dashboard (5 pages)
├── config.yaml                     # paths, target mapping, features, split/CV, risk rule
├── requirements.txt                # pinned package versions
├── pytest.ini
├── .streamlit/config.toml          # turns off Streamlit usage statistics
├── data/
│   ├── raw/cleaned_development_program_dataset.csv   # original file, never modified
│   └── processed/                  # train.csv, test.csv, split_info.json
├── src/
│   ├── utils/config.py             # config loading, relative paths, logging
│   ├── data/                       # load.py, cleaning.py, feature_eligibility.py, audit.py
│   ├── preprocessing/prepare.py    # modelling dataset + 80:20 split
│   ├── eda/                        # eda.py, fy_cutoff_check.py
│   ├── features/selection.py       # correlation, information gain, chi-square
│   ├── models/                     # registry.py (5 models + ensemble), train.py (train/summary/evaluate)
│   ├── evaluation/metrics.py       # metrics, confusion matrices
│   ├── explainability/shap_explain.py
│   ├── decision_support/           # risk.py, recommendations.py, policy.py (Stage 13)
│   ├── prediction/predictor.py     # validate inputs -> prediction, risk, SHAP, recommendations
│   └── app/data.py                 # read-only loaders used by the dashboard
├── artifacts/
│   ├── models/                     # EBM, RuleFit, GPC, QDA, Ensemble (.joblib)
│   ├── cv/                         # CV records + out-of-fold predictions per model
│   ├── shap/background.csv
│   ├── selected_features.json
│   └── model_metadata.json         # selected model, class order, features, versions
├── reports/
│   ├── data_audit/                 # audit report, feature-eligibility register
│   ├── figures/                    # EDA, feature selection, evaluation, SHAP charts
│   ├── logs/                       # run logs (kept as audit evidence; not in the release zip)
│   ├── eda_report.md, fy1990_check.md, feature_selection.md
│   ├── phase3_training_summary.md, model_comparison.md/.csv
│   ├── shap_explanation.md, risk_rules.md, policy_portfolio.md
│   └── quality_audit.md            # final file-by-file audit + accuracy investigation
└── tests/                          # test_phase0_data.py ... test_phase9_release.py (9 files)
```

`artifacts/models/GPC.joblib` is about 97 MB and was left out of the delivered zip because of an upload limit. The dashboard does not need it (see section 9).

---

## 4. Architecture status (14 stages)

| # | Stage | Status | Implementation | Notes / limitations |
|---|---|---|---|---|
| 0 | Historical dataset | Complete | `src/data/audit.py` | Hidden upstream fill values found and handled (section 5) |
| 1 | Data preprocessing | Complete | `cleaning.py`, `prepare.py`, model pipelines | Missing values, duplicates, encoding, scaling. No separate outlier-detection step (section 10). |
| 2 | Exploratory data analysis | Complete | `src/eda/` | Training set only |
| 3 | Feature selection | Complete | `src/features/selection.py` | Run once on the full training set, not inside each CV fold |
| 4 | Train–test split | Complete | `prepare.py` | Stratified 80:20; one row per project, so no project is in both sets |
| 5 | Model training | **Partial** | `src/models/` | EBM, RuleFit, GPC and QDA trained. **TabPFN not trained: weights unavailable** (section 11). GPC used a 2,000-row subsample. |
| 6 | Model evaluation | Complete | `train.py evaluate`, `metrics.py` | Test set used once, after selection |
| 7 | Performance comparison | Complete | `reports/model_comparison.md` | EBM selected by training-set CV macro-F1 |
| 8 | Explainable AI (SHAP) | Complete | `shap_explain.py` | Exact SHAP on EBM |
| 9 | Prediction and risk analysis | Complete | `predictor.py`, `risk.py` | Risk thresholds are provisional |
| 10 | Decision-support recommendations | Complete | `recommendations.py` | Rule-based, non-causal wording |
| 11 | Streamlit web application | Complete (not verified on Windows) | `app.py` | Headless tests and Linux server smoke test only |
| 12 | AI decision-support dashboard | Complete (not verified on Windows) | `app.py` pages 1–4 | |
| 13 | Policy decision support | Complete (not verified on Windows) | `policy.py`, `app.py` page 5 | Uses provisional segment rules; CSV upload tested at module level only |

---

## 5. Data and target

- **Data:** World Bank IEG project evaluations: 12,932 rows, 35 columns, approval FY 1948–2019.
- **Target** (a project-specific three-class grouping, not an official IEG convention):

| IEG rating | Class |
|---|---|
| Highly Satisfactory, Satisfactory | Successful |
| Moderately Satisfactory, Moderately Unsatisfactory | Partially Successful |
| Unsatisfactory, Highly Unsatisfactory | Unsuccessful |

`Not Rated` (162), `Not Available` (9) and `Not Applicable` (9) are excluded.

**Key data decisions (Phase 0; evidence in `reports/data_audit/`):**
1. **Repeated Project IDs are separate IEG evaluations of the same project.** The most recent rated evaluation is kept, leaving 10,750 projects.
2. **The "cleaned" file hides missing values behind constant fill values:** `Lending Project Cost` (42.7% of rows), `ERR at Appraisal` (65.8%) and `ERR at Completion` (69.9%).
3. **Excluded features:**
   - **Net Commitment:** reflects post-approval changes.
   - **ERR at Appraisal:** unreliable because of the fill values.
   - **All post-completion and IEG rating fields, identifiers and URLs.**
   - **Country Code:** corrupted for Namibia; Country Name is used instead.
4. **Final features (7):** Country Name, Sector Board, Agreement Type, Lending Instrument, Approval FY, Lending Project Cost, and `lending_cost_reported` (a yes/no flag).

---

## 6. Models and results

| Model | Setup | Imbalance handling |
|---|---|---|
| EBM | Raw features, native categories and missing values, main effects only | Balanced sample weights |
| RuleFit | One-hot categories + log cost; library is binary-only, so 3 one-vs-rest models | — |
| GPC | Target-encoded categories + scaled numerics; **2,000-row subsample** | Stratified subsample |
| QDA | Same compact encoding; `reg_param` = 0.01 chosen by a CV grid | Equal class priors |
| TabPFN | Code ready; loads v2 weights from a local file | **Not run: weights unavailable** |
| Ensemble (extra) | Equal-weight average of EBM, RuleFit and QDA probabilities | — |

**Results on the held-out test set (2,150 projects).** The model was selected by 5-fold CV on the training set before the test set was used.

| Model | CV macro-F1 | Test macro-F1 | Accuracy | Unsuccessful recall |
|---|---|---|---|---|
| **EBM (selected)** | 0.532 | **0.540** | 0.557 | **0.545** |
| Ensemble | 0.521 | 0.542 | 0.602 | 0.256 |
| QDA | 0.475 | 0.480 | 0.499 | 0.621 |
| GPC | 0.462 | 0.449 | 0.588 | 0.030 |
| RuleFit | 0.448 | 0.448 | 0.599 | 0.020 |
| TabPFN | — | — | — | — |

**Why EBM:**
- It has the best CV score.
- It is the only model that recognises all three classes reasonably well.

The ensemble's test macro-F1 is 0.002 higher than EBM's, but the agreed rule selects on CV, so EBM stays selected. For reference, random guessing in the class proportions scores 0.326 macro-F1 in CV. Full reports are in `reports/model_comparison.md`.

---

## 7. Explainability, risk and recommendations

- **SHAP** (`reports/shap_explanation.md`):
  - Exact SHAP values on EBM's class scores for 500 test projects.
  - The most influential features are Approval FY, Country Name and `lending_cost_reported`.
  - SHAP shows what the model relies on, not what causes outcomes.
- **Risk rules** (provisional, from `config.yaml`):

| Level | Rule |
|---|---|
| High | EBM predicts Unsuccessful |
| Medium | EBM predicts Partially Successful, or Successful with a score below 0.50 |
| Low | EBM predicts Successful with a score of at least 0.50 |

  - The risk score is 1 − score(Successful), used for ranking only. It is not a failure probability.
  - On training out-of-fold predictions, High-risk projects had the highest Unsuccessful share (35%), yet 49% still succeeded. **High means "review carefully", not "will fail".**
- **Recommendations** (`recommendations.py`):
  - Fixed actions for each risk level.
  - Up to three notes on the features that SHAP shows raised the project's risk score.
  - A human-review disclaimer on every result.

---

## 8. Stage 13: Policy decision support

A portfolio-level view for planning and resource allocation, separate from the per-project recommendations. Report: `reports/policy_portfolio.md`. Dashboard: the **Policy decision support** page.

- **Data source:** the EBM's **out-of-fold predictions on the 8,600 training projects** (`artifacts/cv/EBM_oof.npy`).
  - Each project was scored by a model that had not seen it during cross-validation.
  - `verify_oof()` checks the file before use: its shape, that every project was scored, and that the per-fold scores match the training record exactly.
  - The test set is not used.
- **Risk summaries:** Low, Medium and High shares by **sector**, **region** and **lending instrument**, using the existing risk rules unchanged. Each segment also shows its historical actual Unsuccessful share.
- **Policy considerations** (rule-based; provisional settings in `policy.py`):
  - Only segments with **at least 30 projects** get considerations; smaller ones are listed as too small.
  - A segment stands out when its share differs from the portfolio share by **at least 10 percentage points**.
  - **Supervision allocation:** prioritise segments with many High-risk projects; standard supervision for segments with many Low-risk projects. Segments with many of both are marked "mixed — allocate case by case".
  - **Risk mitigation:** readiness reviews, early-warning indicators and mid-term reviews for high-risk concentrations.
  - **Planning cautions:** patterns are associations, not causes, and small segments are flagged.
- **CSV upload (optional):**
  1. Download the template.
  2. Fill one row per proposed project. Required columns: Country Name, Sector Board, Agreement Type, Lending Instrument, Approval FY. Optional: Lending Project Cost (USD), Region.
  3. Upload it. Every row is validated and scored with the same model and risk rules as the prediction page.
  4. Invalid rows are listed with reasons and never scored.
  5. The scored portfolio can be downloaded.

---

## 9. When a model file is missing

`python -m src.models.train evaluate` checks every trained model's `.joblib` file before evaluating.

- **A file is missing (for example `GPC.joblib`):**
  - A message names the file and explains how to restore it: copy the original back into `artifacts/models/`, or rebuild it with `python -m src.models.train train --models GPC` (about 7 minutes).
  - That model is skipped. The others are evaluated normally.
  - The report lists the skipped model under "Not evaluated in this run (model file missing)".
- **The CV-selected model's file is missing (EBM):** evaluation stops with a clear error before writing any report.
- **What never happens:** models are not retrained automatically, and the selection rule is unchanged, because CV scores come from the saved CV records.

The dashboard needs only `EBM.joblib`, the CV files and the reports, so it works without `GPC.joblib`.

---

## 10. How cost outliers are handled

There is **no separate outlier detection or removal step**. The code does the following:
- **Invalid values are repaired or flagged** (Phase 0):
  - One impossible `Approval FY` (0) is repaired from the approval date.
  - The upstream fill value for project cost is converted back to "missing" and flagged with `lending_cost_reported`.
- **Extreme project costs are kept.** Costs range from $0.19M to about $29.8B.
  - For QDA, GPC, RuleFit and TabPFN, cost is **log10-transformed** and then scaled. This reduces the influence of very large values.
  - EBM uses cost directly, through its own built-in binning of numeric features (the library default).
- **The dashboard** rejects non-positive costs and warns when a cost is outside the training range.

---

## 10a. Final quality audit and the 95% accuracy question

The full audit is in `reports/quality_audit.md`. In summary:
- **No leakage or evaluation flaw was found.** The checks covered: one evaluation per project, a reproducible split with no overlap, all fitting on training data only, approval-time features only, and genuine out-of-fold predictions.
- **Two defects were fixed, neither of which changes any results:**
  - The Model performance page hard-coded result numbers; it now builds them from the saved results.
  - An unused import was removed.
- **Accuracy experiments** (training-set CV only; the test set was not used):
  - Six candidates, including tuned EBM variants and two reference models (gradient boosting, logistic regression), all reach **0.53–0.54 macro-F1**.
  - Removing class weights raises accuracy to 60.5%, but only 12% of Unsuccessful projects are then found.
- **95% accuracy is not achievable with valid evaluation on this data.**
  - When IEG rated the same project twice, the three-class labels agreed only **71.7%** of the time.
  - Within each approval era, macro-F1 is only about 0.41.
  - The production EBM (test macro-F1 0.540) remains the best valid, well-balanced result.
- **Not promoted:** an EBM with pairwise interactions improved CV macro-F1 by +0.009 (about one standard deviation). That gain is within noise, so it was **not** promoted, and production models and results are unchanged.

Experiment scripts and raw outputs are kept in `experiments/`, which is excluded from the release zip.

## 11. Limitations

- **TabPFN was not trained.** Its pre-trained weights could not be downloaded in the development environment, and newer versions require a licence login. The code is ready. To add it:
  1. Download `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` from <https://huggingface.co/Prior-Labs/TabPFN-v2-clf>.
  2. Place it in `artifacts\tabpfn\`.
  3. Run `python -m src.models.train train --models TabPFN`, then `evaluate`.

  Loading v2 weights with the installed `tabpfn` 9.2 is untested.
- **Thresholds are provisional.** The 0.50 risk cut-off and the policy settings (30 projects, 10 points) are project choices, not validated decision criteria.
- **The rating scale changed over time.** "Partially Successful" is rare before FY1990 (7% vs 49% after), so Approval FY partly reflects rating practice, not just project differences (`reports/fy1990_check.md`).
- **Association, not causation.** SHAP values, risk levels and policy patterns describe what the model relies on and historical associations. They do not show what causes outcomes.
- **Scores are not calibrated probabilities.** EBM was trained with balanced weights, so its scores lean towards the smaller classes.
- **Performance is moderate.** Test macro-F1 is 0.54 from approval-time information only.
- **Smaller limitations:** GPC used a 2,000-row subsample; feature selection was not repeated inside each CV fold.
- **Not verified on Windows or in a browser.** No step has been run on Windows, the dashboard has not been opened in a real browser, and the CSV upload has not been tried through the UI.

---

## 12. Testing and verification status

Run all tests with `python -m pytest -q`. They need no training and no downloads, and they never modify the saved models or reports.

| Test file | Covers |
|---|---|
| `test_phase0_data.py` | Schema, target mapping, record selection, feature-eligibility rules |
| `test_phase1_3.py` | Split integrity, selected features, every model's output format (tiny throwaway fits) |
| `test_phase4_leakage.py` | No test-set use in training, feature selection, tuning or selection |
| `test_phase5_shap.py` | SHAP output shape, additivity, approval-time features only |
| `test_phase6_decision_support.py` | Risk rules, recommendation rules, input validation |
| `test_phase7_app.py` | All dashboard pages render; form success and error paths; missing model handled |
| `test_phase8_evaluate_missing.py` | `evaluate` with a missing model file (runs on a temporary copy) |
| `test_phase8_policy.py` | Out-of-fold verification, portfolio summaries, policy rules, CSV scoring, policy page |
| `test_phase9_release.py` | Dashboard result captions come from saved results, not hard-coded numbers |

**Verified in the final audit (Linux, Python 3.11.17):**
- `python -m pytest -q`: **70 passed, 1 skipped, 0 failed**. The skip is TabPFN (weights unavailable). The warnings are library deprecation notices.
- All five dashboard pages ran through Streamlit's test runner with no exceptions and no error messages.
- A fresh `streamlit run` server returned health `ok` and HTTP 200, with no errors in its log.
- Stage 13 (verified in Phase 8E): out-of-fold file verified, test set not used, invalid CSV rows rejected with reasons.
- Example predictions and the 0.50 risk threshold are unchanged. `evaluate` handles a missing `GPC.joblib`.
- `requirements.txt` resolves for Windows with Python 3.12 and 3.11 (dependency resolution only, nothing installed on Windows).

**Not yet verified:**
- Anything on Windows.
- Running on Python 3.12 (only a dependency dry run).
- Viewing the dashboard in a browser.

---

## 13. HOD demonstration walkthrough (about 10 minutes)

1. **Start:** run `streamlit run app.py`. Explain that it loads saved models and does not train anything.
2. **Predict a project:**
   - Choose Kenya, Water, IDA, SIL, FY2010, $60M, then click **Predict**.
   - Show the predicted outcome, the **Medium** risk level and the recommended actions.
   - Point out the "estimate, not a guarantee" note.
   - Submit with an empty field to show the validation message.
3. **Risk classification:** explain the three rules in section 7, and that High means "review carefully".
4. **Explainability:**
   - Show the global SHAP chart: approval year, country and cost-reporting matter most.
   - Show the explanation of the prediction you just made: red bars raised the risk score, blue bars lowered it.
   - Say that this shows what the model relies on, not causes.
5. **Historical insights:** show the outcome mix by decade, mention the FY1990 rating-scale note, and look up a country. This is past data, kept separate from predictions.
6. **Model performance:**
   - Show the comparison table and confusion matrices.
   - EBM was selected by training-set CV.
   - RuleFit and GPC look accurate only because they almost never predict "Unsuccessful".
   - TabPFN is honestly marked as unavailable.
7. **Policy decision support:**
   - Open the Region tab. AFR has 40% High-risk projects against a 26% average, so the page suggests prioritising supervision there.
   - Point out the 30-project and 10-point rules and the association caution.
   - Optionally, download the template, upload it, and show the scored portfolio.
