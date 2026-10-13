# Phase 1 — Exploratory Data Analysis (training set only)

Training projects: 8,600. All figures are in `reports/figures/`.

## Class balance

- Successful: 4,253 (49.5%)
- Partially Successful: 2,757 (32.1%)
- Unsuccessful: 1,590 (18.5%)

The classes are imbalanced (Unsuccessful is the smallest), so macro-F1 is used to compare models: it gives every class equal weight.

## Time trend

| Decade | Successful | Partially | Unsuccessful |
|---|---|---|---|
| 1940s | 100% | 0% | 0% |
| 1950s | 100% | 0% | 0% |
| 1960s | 86% | 0% | 14% |
| 1970s | 75% | 0% | 24% |
| 1980s | 56% | 13% | 31% |
| 1990s | 50% | 34% | 16% |
| 2000s | 30% | 60% | 10% |
| 2010s | 35% | 56% | 9% |

## Region and instrument

- Highest success share by region: EAP (62%); lowest: AFR (39%).
- DPF: 42% Successful
- IPF: 51% Successful
- PforR: 100% Successful

## Project cost

- Median cost, Successful: $65.6M
- Median cost, Partially Successful: $50.0M
- Median cost, Unsuccessful: $45.0M
- Success share when cost reported: 38%; when not reported: 65% (unreported cost is mostly older projects, so this partly reflects era).

## Notes

- These are associations in historical data, not causes.
- Feature selection (next phase) tests which of these features carry predictive signal.