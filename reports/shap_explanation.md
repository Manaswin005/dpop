# Phase 5 — Explaining the EBM model with SHAP

Method: exact SHAP values on predicted class probabilities, 500 test projects, 100 training projects as background. Additivity check: max error 2.3e-15 (contributions add up exactly to the model's probabilities).

## 1. Global importance (`shap_01_global_importance.png`)

| Feature | Successful | Partially Successful | Unsuccessful | Total |
|---|---|---|---|---|
| Approval FY | 0.121 | 0.166 | 0.064 | 0.351 |
| Country Name | 0.066 | 0.029 | 0.072 | 0.167 |
| lending_cost_reported | 0.029 | 0.078 | 0.048 | 0.156 |
| Sector Board | 0.041 | 0.013 | 0.045 | 0.098 |
| Lending Project Cost | 0.034 | 0.030 | 0.011 | 0.075 |
| Lending Instrument | 0.024 | 0.017 | 0.023 | 0.064 |
| Agreement Type | 0.024 | 0.016 | 0.010 | 0.050 |

**How to read it:**
- Each bar shows how much a feature moves the predicted probabilities, on average, across the 500 projects.
- A value of 0.10 means the feature typically shifts a class probability by about 10 percentage points.
- The coloured segments show which outcome class the shift affects.
- The most influential features are **Approval FY**, **Country Name** and **lending_cost_reported**.

## 2. One project explained (`shap_02_local_example.png`)

The model predicts this project as Unsuccessful with probability 0.72 (true outcome: Unsuccessful). It starts from the average P(Unsuccessful) of 0.34:

| Feature | Value | Change in P(Unsuccessful) |
|---|---|---|
| Country Name | Tanzania | +0.150 |
| Approval FY | 1980 | +0.130 |
| Lending Instrument | FIL | +0.071 |
| lending_cost_reported | no | +0.057 |
| Sector Board | Financial Sector | -0.018 |
| Agreement Type | IBRD | -0.008 |
| Lending Project Cost | not reported | +0.003 |

**How to read it:**
- Start at the bottom value E[f(x)], the average prediction.
- Red bars push the probability of Unsuccessful up; blue bars push it down.
- The top value f(x) is this project's final probability.

## 3. Cautions

- SHAP shows what the model **relies on** to make predictions. It does **not** show causes. For example, an approval year that raises the predicted risk does not mean the year caused the outcome.
- Approval FY, Lending Project Cost and lending_cost_reported are related (older projects rarely have a reported cost). The model's credit can be split between them, so read them together.
- Approval FY partly reflects the change in rating scale before FY1990 (see README).
- EBM was trained with balanced class weights, so its probabilities are tilted towards the smaller classes. That is why the average P(Unsuccessful) is above the 18.5% share in the data. These are model scores, not calibrated real-world probabilities.
- Only approval-time features are used. No post-completion information is an input.