# Model comparison

Winner chosen by **training-set 5-fold CV macro-F1**. Test metrics are reported afterwards and do not change the choice.

| model                  |   cv_macro_f1 |   cv_std |   test_macro_f1 |   test_accuracy |   test_weighted_f1 |   test_roc_auc_ovr_macro |   train_rows_used | selected   |
|:-----------------------|--------------:|---------:|----------------:|----------------:|-------------------:|-------------------------:|------------------:|:-----------|
| EBM                    |        0.5318 |   0.0098 |          0.5404 |          0.5567 |             0.5503 |                   0.7579 |              8600 | True       |
| Ensemble (soft voting) |        0.5215 |   0.0111 |          0.5424 |          0.6023 |             0.5887 |                   0.7569 |              8600 | False      |
| QDA                    |        0.475  |   0.0086 |          0.48   |          0.4991 |             0.4653 |                   0.7414 |              8600 | False      |
| GPC                    |        0.4616 |   0.0141 |          0.4489 |          0.5884 |             0.5376 |                   0.7383 |              2000 | False      |
| RuleFit                |        0.4477 |   0.0145 |          0.4477 |          0.5986 |             0.5426 |                   0.7412 |              8600 | False      |

**Not trained:**
- **TabPFN**: BLOCKED — TabPFN weights not found at /home/claude/development-program-outcome-prediction/artifacts/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt. Download tabpfn-v2-classifier-finetuned-zk73skhh.ckpt from https://huggingface.co/Prior-Labs/TabPFN-v2-clf and place it there (see README).

**Notes:**
- RuleFit: imodels RuleFit is binary-only: three one-vs-rest RuleFit models, probabilities normalised.
- GPC: Trained on a stratified subsample of the training set (GPC cost grows with n^3).
- Ensemble (soft voting): Equal-weight average of EBM, RuleFit, QDA probabilities.

# Classification reports (test set)

## EBM

```
                      precision    recall  f1-score   support

          Successful      0.751     0.400     0.522      1063
Partially Successful      0.553     0.806     0.656       689
        Unsuccessful      0.373     0.545     0.443       398

            accuracy                          0.557      2150
           macro avg      0.559     0.584     0.540      2150
        weighted avg      0.618     0.557     0.550      2150

```

## RuleFit

```
                      precision    recall  f1-score   support

          Successful      0.603     0.767     0.675      1063
Partially Successful      0.590     0.673     0.629       689
        Unsuccessful      0.667     0.020     0.039       398

            accuracy                          0.599      2150
           macro avg      0.620     0.487     0.448      2150
        weighted avg      0.611     0.599     0.543      2150

```

## GPC

```
                      precision    recall  f1-score   support

          Successful      0.613     0.700     0.654      1063
Partially Successful      0.559     0.739     0.636       689
        Unsuccessful      0.462     0.030     0.057       398

            accuracy                          0.588      2150
           macro avg      0.545     0.490     0.449      2150
        weighted avg      0.568     0.588     0.538      2150

```

## QDA

```
                      precision    recall  f1-score   support

          Successful      0.798     0.230     0.358      1063
Partially Successful      0.530     0.843     0.651       689
        Unsuccessful      0.331     0.621     0.432       398

            accuracy                          0.499      2150
           macro avg      0.553     0.565     0.480      2150
        weighted avg      0.626     0.499     0.465      2150

```

## Ensemble (soft voting)

```
                      precision    recall  f1-score   support

          Successful      0.673     0.613     0.642      1063
Partially Successful      0.565     0.785     0.657       689
        Unsuccessful      0.455     0.256     0.328       398

            accuracy                          0.602      2150
           macro avg      0.565     0.552     0.542      2150
        weighted avg      0.598     0.602     0.589      2150

```
