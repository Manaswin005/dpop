"""Phase 3 — Model definitions (architecture Stage 5).

Every model is a single sklearn-compatible object that takes the raw feature
DataFrame and does its own preprocessing inside. So each CV fold and the final
fit learn preprocessing from training rows only, and the saved file contains
everything needed for prediction.

Model-specific preprocessing
  EBM      raw features; EBM handles categories and missing values natively.
  RuleFit  one-hot categories (rare levels grouped) + log cost, so rules read
           like "Sector = Education". Binary-only library -> one-vs-rest x3.
  QDA, GPC, TabPFN  compact numeric matrix: each category is replaced by the
           per-class outcome rate learned from training data (target encoding
           with internal cross-fitting), plus scaled year and log cost.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.utils.config import load_config, resolve

COST = "Lending Project Cost"
MODEL_NAMES = ["EBM", "TabPFN", "RuleFit", "GPC", "QDA"]


class ModelUnavailable(RuntimeError):
    """A required model cannot run in this environment (reason in message)."""


# ----------------------------------------------------------------------------
# Shared preprocessing pieces
# ----------------------------------------------------------------------------
def _log_cost(x):
    return np.log10(np.asarray(x, dtype=float))


def _numeric_block(nums: list[str]) -> list:
    """Year: scale. Cost: log10 -> median impute (training median) -> scale. Flag: as is."""
    blocks = []
    if "Approval FY" in nums:
        blocks.append(("year", StandardScaler(), ["Approval FY"]))
    if COST in nums:
        blocks.append(("cost", make_pipeline(FunctionTransformer(_log_cost, feature_names_out="one-to-one"),
                                             SimpleImputer(strategy="median"), StandardScaler()), [COST]))
    if "lending_cost_reported" in nums:
        blocks.append(("cost_flag", "passthrough", ["lending_cost_reported"]))
    return blocks


def compact_encoder(cats, nums, seed) -> ColumnTransformer:
    """Dense, low-dimensional representation for QDA / GPC / TabPFN."""
    te = TargetEncoder(target_type="multiclass", random_state=seed)
    return ColumnTransformer([("cats", te, cats), *_numeric_block(nums)], verbose_feature_names_out=False)


def onehot_encoder(cats, nums) -> ColumnTransformer:
    """Readable one-hot representation for RuleFit (rare levels -> 'infrequent')."""
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)
    return ColumnTransformer([("cats", ohe, cats), *_numeric_block(nums)], verbose_feature_names_out=False)


# ----------------------------------------------------------------------------
# Small wrappers
# ----------------------------------------------------------------------------
class BalancedEBM(BaseEstimator, ClassifierMixin):
    """EBM trained with class-balanced sample weights (helps the small classes)."""

    def __init__(self, cats=None, nums=None, seed=42):
        self.cats, self.nums, self.seed = cats, nums, seed

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingClassifier
        cols = list(self.cats) + list(self.nums)
        self.model_ = ExplainableBoostingClassifier(
            feature_names=cols,
            feature_types=["nominal"] * len(self.cats) + ["continuous"] * len(self.nums),
            interactions=0,          # main effects only: one shape function per feature
            random_state=self.seed, n_jobs=-1,
        )
        self.model_.fit(X[cols], y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[list(self.cats) + list(self.nums)])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class SubsampleClassifier(BaseEstimator, ClassifierMixin):
    """Fit the wrapped estimator on a stratified subsample of the TRAINING rows.

    Used for GPC, whose training cost grows with the cube of the row count.
    """

    def __init__(self, estimator=None, n_samples=2000, seed=42):
        self.estimator, self.n_samples, self.seed = estimator, n_samples, seed

    def fit(self, X, y):
        y = np.asarray(y)
        if len(y) > self.n_samples:
            idx, _ = train_test_split(np.arange(len(y)), train_size=self.n_samples,
                                      stratify=y, random_state=self.seed)
            X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            y = y[idx]
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.n_train_used_ = len(y)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def _tabpfn():
    """TabPFN v2 loaded from a local weights file (no network needed at run time)."""
    cfg = load_config()
    path = resolve(cfg["models"]["tabpfn_model_path"])
    if not path.exists():
        raise ModelUnavailable(
            f"TabPFN weights not found at {path}. Download "
            "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt from "
            "https://huggingface.co/Prior-Labs/TabPFN-v2-clf and place it there (see README).")
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(model_path=str(path), device="cpu", n_estimators=4,
                            balance_probabilities=True, random_state=cfg["project"]["random_seed"])


# ----------------------------------------------------------------------------
# Public factory
# ----------------------------------------------------------------------------
def build_model(name: str, cats: list[str], nums: list[str]):
    """Return an unfitted model (preprocessing included) by name."""
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    n_classes = len(cfg["target"]["class_order"])

    if name == "EBM":
        return BalancedEBM(cats=cats, nums=nums, seed=seed)
    if name == "QDA":
        # Equal priors counter the class imbalance; reg_param stabilises the
        # per-class covariance matrices (value chosen by small CV grid in training).
        return Pipeline([("prep", compact_encoder(cats, nums, seed)),
                         ("clf", QuadraticDiscriminantAnalysis(priors=np.full(n_classes, 1 / n_classes), reg_param=0.1))])
    if name == "GPC":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gpc = GaussianProcessClassifier(kernel=kernel, multi_class="one_vs_rest", random_state=seed, n_jobs=-1)
        return SubsampleClassifier(Pipeline([("prep", compact_encoder(cats, nums, seed)), ("clf", gpc)]),
                                   n_samples=cfg["models"]["gpc_subsample"], seed=seed)
    if name == "RuleFit":
        from imodels import RuleFitClassifier
        rf = RuleFitClassifier(max_rules=30, random_state=seed)
        return Pipeline([("prep", onehot_encoder(cats, nums)), ("clf", OneVsRestClassifier(rf))])
    if name == "TabPFN":
        return Pipeline([("prep", compact_encoder(cats, nums, seed)), ("clf", _tabpfn())])
    raise ValueError(f"Unknown model: {name}")


class AveragedEnsemble(BaseEstimator, ClassifierMixin):
    """Equal-weight soft-voting ensemble of already-fitted models.

    Prediction = average of the members' class probabilities. No weights or
    meta-model are learned, so nothing can be fitted on test data. Its CV score
    is the average of the members' out-of-fold probabilities on the same folds.
    """

    def __init__(self, members: dict | None = None):
        self.members = members or {}

    def fit(self, X=None, y=None):  # members are fitted individually beforehand
        self.classes_ = next(iter(self.members.values())).classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members.values()], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
