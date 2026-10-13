# Phase 6 — Risk rules

| Risk level | Rule |
|---|---|
| **High** | Model predicts Unsuccessful |
| **Medium** | Model predicts Partially Successful, or predicts Successful with score < 0.50 |
| **Low** | Model predicts Successful with score ≥ 0.50 |

The 0.50 cut-off is a **provisional, project-defined** choice. It means the model gives Successful at least half of its total score. It has not been validated against any external decision criterion.

## What the levels looked like on the training set

These come from the EBM's out-of-fold predictions: each project was scored by a model that had not seen it during training. The test set was not used.

| Risk level | Projects | Actually Successful | Actually Partially | Actually Unsuccessful |
|---|---|---|---|---|
| Low | 1,741 | 80% | 5% | 15% |
| Medium | 4,628 | 38% | 50% | 12% |
| High | 2,231 | 49% | 16% | 35% |

How to read this:
- **Low** projects were mostly Successful (80%).
- **High** projects had the largest Unsuccessful share (35%, against 12%–15% elsewhere). But 49% of them still turned out Successful. The model over-flags on purpose (balanced class weights), so High means **'review carefully'**, not 'will fail'.
- **Medium** projects were mostly Partially Successful or Successful.