# DPOP — Final quality audit and accuracy investigation

All checks were run in the development workspace (Linux, Python 3.11.17). Experiment scripts and raw results are kept separately in `experiments/`. They are not part of the release.

## 1. File-by-file audit

| File | Purpose | Used by | Finding | Action |
|---|---|---|---|---|
| `app.py` | Streamlit dashboard, 5 pages | Demo | Units correct (USD millions × 1,000,000 → USD; no INR). **Defect:** result numbers hard-coded in captions (0.542/0.540, 2,150, 8,600, 2,000-row) | Fixed: built from saved results |
| `src/app/data.py` | Read-only loaders for the app | `app.py` | OK | Added `performance_notes()` and `held_out_size()` |
| `src/utils/config.py` | Config, relative paths, logging | All modules | OK | None |
| `src/data/load.py` | Load CSV + 35-column schema check | Audit, prepare | OK | None |
| `src/data/cleaning.py` | Stateless pre-split steps | Audit, prepare | OK (verified reproducible) | None |
| `src/data/feature_eligibility.py` | Decision for every column | Audit, tests | OK (all used features are INCLUDE) | None |
| `src/data/audit.py` | Stage 0 report | Rebuild only | OK | None |
| `src/preprocessing/prepare.py` | Modelling data + stratified 80:20 split | All later stages | OK (re-split reproduces saved files exactly) | None |
| `src/eda/eda.py`, `fy_cutoff_check.py` | Stage 2 charts, era check | Rebuild, app images | OK (training set only) | None |
| `src/features/selection.py` | Stage 3: correlation, information gain, chi-square | Rebuild | OK (training set only; reproducibility test passes) | None |
| `src/models/registry.py` | 5 models + ensemble definitions | Train, predictor | **Defect (minor):** unused `pandas` import | Fixed |
| `src/models/train.py` | CV training, summary, evaluation | Rebuild, tests | OK, including the missing-file handling | None |
| `src/evaluation/metrics.py` | Metrics, confusion matrices | `train.py` | OK | None |
| `src/explainability/shap_explain.py` | Stage 8 exact SHAP | Rebuild, predictor | OK (additivity verified) | None |
| `src/decision_support/risk.py` | Risk rules (0.50 cut-off) | Predictor, policy | OK, unchanged | None |
| `src/decision_support/recommendations.py` | Per-project recommendations | Predictor | OK | None |
| `src/decision_support/policy.py` | Stage 13 portfolio view | App page 5 | OK (out-of-fold verified, never reads the test set) | None |
| `src/prediction/predictor.py` | Validate → predict → risk → SHAP → recommendations | App, policy | OK (example outputs unchanged) | None |
| `tests/` (9 files) | Regression tests | `pytest` | OK | Added `test_phase9_release.py` |
| `config.yaml` | Paths, mapping, features, risk rule | All | OK | None |
| `requirements.txt` | Pinned versions | Setup | Resolves for **Windows + Python 3.12 and 3.11** (dependency resolution only; not installed on Windows) | None |
| `README.md` | Documentation | Users | Needed the audit and accuracy findings | Updated |
| `.gitignore`, `.streamlit/config.toml` | Git rules, Streamlit setting | Setup | OK (`.gitignore` hides `artifacts/` and `data/processed/`; relevant only if Git is used) | None |
| `data/raw/*.csv` | Original data | Pipeline | 12,932 × 35; 0 exact duplicates; hidden fill values (documented in Phase 0) | None |
| `data/processed/*` | Split | All stages | 8,600 / 2,150 projects, no overlap, reproducible | None |
| `artifacts/*` | Models, CV records, out-of-fold predictions, SHAP background, metadata | App, evaluation | All present in the workspace; `GPC.joblib` is excluded from the zip (size) | None |
| `reports/*` | Generated reports and charts | App, documentation | Consistent with the code | Added this file |

Static checks: every file compiles; `pyflakes` found only the unused import above; there are no defined-but-unused functions.

## 2. Dependency map

```
data/raw CSV
  -> src/data (load, cleaning, eligibility, audit)                [Stage 0-1]
  -> src/preprocessing/prepare -> data/processed/train.csv, test.csv  [Stages 1, 4]
       -> src/eda (training only) -> reports/figures                [Stage 2]
       -> src/features/selection (training only) -> artifacts/selected_features.json   [Stage 3]
       -> src/models/train  train    -> artifacts/models, artifacts/cv (CV + out-of-fold)   [Stage 5]
                            evaluate -> reports/model_comparison.*, artifacts/model_metadata.json   [Stages 6-7]
       -> src/explainability/shap_explain -> SHAP charts, artifacts/shap/background.csv   [Stage 8]
  -> src/prediction/predictor (EBM + metadata + SHAP + risk + recommendations)   [Stages 9-10]
  -> src/decision_support/policy (verified EBM out-of-fold + risk rules)   [Stage 13]
  -> app.py (reads predictor, reports, figures, policy)   [Stages 11-13]
```

## 3. Data and leakage findings

| Check | Result |
|---|---|
| Raw data | 12,932 rows × 35 columns, 0 missing cells (fill values hide the true gaps), 0 exact duplicates |
| One evaluation per project | Yes: 10,750 unique projects |
| Split reproducible | Yes: re-running the split gives identical files |
| Project overlap between train and test | 0 |
| Class balance, train vs test | Identical within 0.02 percentage points |
| Fitting on test data | None: imputation, encoding, scaling and target encoding are fitted inside each model's training call; feature selection and tuning use training data only |
| Feature eligibility | All 7 features are approval-time (INCLUDE); no IEG rating, post-completion field, ID or URL is used |
| Target proxy check | No single feature predicts the label well. The strongest, Approval FY alone, reaches 59.5% accuracy on training data (most common class per year band). |
| CV design | Stratified 5-fold (folds differ from overall class shares by at most 0.03 points); one row per project, so no grouping is needed |
| Out-of-fold predictions | Genuine: refitting fold 1 reproduces them exactly, and the fold scores match the training record |

**Distribution shift and confounding (training out-of-fold predictions):**

| Era | Projects | Class mix (S / P / U) | Macro-F1 | Accuracy | Predicted "Partially" |
|---|---|---|---|---|---|
| Before FY1990 | 3,457 | 66% / 7% / 28% | 0.403 | 0.579 | 0.5% |
| FY1990 onward | 5,143 | 39% / 49% / 12% | 0.410 | 0.535 | 78% |

Overall macro-F1 is 0.53, but within each era it is only about 0.41. Much of the model's skill comes from telling eras apart, which partly reflects how IEG rating practice changed. Approval FY is a legitimate approval-time feature, but this confounding must be stated when interpreting results.

**Label reliability:** for 1,995 projects that IEG rated more than once, the three-class label agreed only **71.7%** of the time. Two expert evaluations of the same project disagree on more than a quarter of projects.

## 4. Accuracy investigation (training-set 5-fold CV, production folds)

| Candidate | CV macro-F1 | Accuracy | Balanced accuracy | F1 S / P / U | Unsuccessful recall |
|---|---|---|---|---|---|
| **EBM baseline (production)** | **0.532 ± 0.010** | 0.553 | 0.568 | 0.54 / 0.65 / 0.41 | 0.49 |
| EBM without class weights | 0.498 ± 0.010 | **0.605** | 0.513 | 0.68 / 0.63 / 0.19 | 0.12 |
| EBM + 10 interactions | 0.541 ± 0.009 | 0.560 | 0.576 | 0.55 / 0.65 / 0.42 | 0.51 |
| EBM + Region | 0.530 ± 0.008 | 0.552 | 0.566 | 0.54 / 0.65 / 0.41 | 0.49 |
| HistGradientBoosting (reference) | 0.544 ± 0.015 | 0.577 | 0.552 | 0.63 / 0.61 / 0.40 | 0.41 |
| Logistic regression (reference) | 0.531 ± 0.004 | 0.550 | 0.561 | 0.55 / 0.64 / 0.40 | 0.50 |

- **The baseline reproduces exactly** (0.5318).
- **Higher accuracy can be bought, but at a cost.** Dropping class weights raises accuracy to 60.5% but finds only 12% of Unsuccessful projects.
- **The best EBM variant (interactions)** improves on the baseline in all 5 folds, but only by +0.009 on average, about one standard deviation. It was also the best of six candidates, so the gain is slightly optimistic.
- **Very different model types converge.** Boosting, logistic regression and EBM all score 0.53–0.54. This points to limited signal in approval-time features, not a weak model.

## 5. Is 95% accuracy achievable?

**No. 95% accuracy is not supported by any valid evaluation, and it is not realistically achievable with this data.**
1. **The models converge.** Every model type tried reaches 0.53–0.54 macro-F1 and 55–61% accuracy under leakage-safe cross-validation.
2. **The labels are noisy.** IEG's own repeated evaluations agree on the three-class label only 71.7% of the time, so a model predicting these labels cannot be expected to be far more consistent than the experts.
3. **The information is limited.** At approval, only 7 descriptive features are available. Nothing is known yet about implementation, and most of the signal reflects approval era.

Higher numbers could be produced only by invalid means: using post-completion IEG ratings as inputs, which is leakage; tuning on the test set; or merging classes. None of these were done.

**Best valid result:** EBM (production), with test macro-F1 0.540, accuracy 0.557 and Unsuccessful recall 0.545. The optional interaction variant would add about +0.01 CV macro-F1. It has **not** been promoted (see the README).
