# Phase 3 — Training summary (5-fold CV on the training set only)

| Model | Status | CV macro-F1 (mean ± std) | Training rows | CV time (s) | Final fit (s) |
|---|---|---|---|---|---|
| EBM | TRAINED | 0.5318 ± 0.0098 | 8,600 | 48.9 | 12.5 |
| TabPFN | BLOCKED | — | — | — | — |
| RuleFit | TRAINED | 0.4477 ± 0.0145 | 8,600 | 196.1 | 46.9 |
| GPC | TRAINED | 0.4616 ± 0.0141 | 2,000 | 362.0 | 74.3 |
| QDA | TRAINED | 0.4750 ± 0.0086 | 8,600 | 0.3 | 0.1 |

Notes:
- TabPFN blocker: TabPFN weights not found at /home/claude/development-program-outcome-prediction/artifacts/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt. Download tabpfn-v2-classifier-finetuned-zk73skhh.ckpt from https://huggingface.co/Prior-Labs/TabPFN-v2-clf and place it there (see README).
- RuleFit: imodels RuleFit is binary-only: three one-vs-rest RuleFit models, probabilities normalised.
- GPC: Trained on a stratified subsample of the training set (GPC cost grows with n^3).
- QDA reg_param grid (CV macro-F1): {'0.01': 0.47499781308176525, '0.05': 0.44757007450644204, '0.1': 0.425472171089386, '0.3': 0.3985855173071979, '0.6': 0.3915304969944617} -> chosen 0.01

No model has been scored on the test set yet (that is Phase 4).