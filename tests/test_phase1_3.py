"""Phase 1–3 tests: split integrity, feature selection output, model plumbing."""
import json

import numpy as np
import pandas as pd
import pytest

from src.models.registry import AveragedEnsemble, ModelUnavailable, build_model
from src.preprocessing.prepare import feature_lists, load_split
from src.utils.config import load_config, resolve

CFG = load_config()
TARGET = CFG["columns"]["target"]


@pytest.fixture(scope="module")
def split():
    return load_split()


@pytest.fixture(scope="module")
def selected():
    sel = json.loads((resolve(CFG["paths"]["artifacts_dir"]) / "selected_features.json").read_text())
    return sel["categorical"], sel["numerical"]


# ---- split ----------------------------------------------------------------
def test_split_sizes_and_no_project_overlap(split):
    train, test = split
    assert len(train) + len(test) == 10750
    assert abs(len(test) / (len(train) + len(test)) - 0.20) < 0.001
    assert not set(train["Project ID"]) & set(test["Project ID"])


def test_split_is_stratified(split):
    train, test = split
    a = train[TARGET].value_counts(normalize=True)
    b = test[TARGET].value_counts(normalize=True)
    assert (a - b).abs().max() < 0.01


def test_processed_files_have_only_allowed_columns(split):
    cats, nums = feature_lists()
    allowed = {"Project ID", TARGET, *cats, *nums}
    for df in split:
        assert set(df.columns) == allowed


# ---- feature selection ----------------------------------------------------
def test_selected_features_are_eligible(selected):
    cats, nums = feature_lists()
    sel_c, sel_n = selected
    assert set(sel_c) <= set(cats) and set(sel_n) <= set(nums)
    assert len(sel_c) + len(sel_n) >= 3


# ---- model plumbing (tiny training samples, fast) -------------------------
@pytest.fixture(scope="module")
def tiny(split, selected):
    train, test = split
    cats, nums = selected
    order = CFG["target"]["class_order"]
    tr = train.groupby(TARGET, group_keys=False).apply(lambda g: g.sample(60, random_state=0))
    y = tr[TARGET].map(order.index).to_numpy()
    return tr[cats + nums], y, test[cats + nums].head(25), cats, nums


@pytest.mark.parametrize("name", ["EBM", "QDA", "RuleFit", "GPC", "TabPFN"])
def test_model_outputs_valid_probabilities(name, tiny):
    X, y, X_new, cats, nums = tiny
    try:
        model = build_model(name, cats, nums)
        if name == "QDA":
            model.set_params(clf__reg_param=0.3)
        model.fit(X, y)
    except ModelUnavailable as e:
        pytest.skip(f"{name} unavailable in this environment: {e}")
    proba = model.predict_proba(X_new)
    assert proba.shape == (len(X_new), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert list(model.classes_) == [0, 1, 2]


def test_unseen_category_does_not_crash(tiny):
    X, y, X_new, cats, nums = tiny
    model = build_model("QDA", cats, nums).set_params(clf__reg_param=0.3).fit(X, y)
    odd = X_new.head(2).copy()
    odd[cats[0]] = "A category never seen in training"
    assert model.predict_proba(odd).shape == (2, 3)


def test_ensemble_is_plain_average():
    class Fixed:
        classes_ = np.array([0, 1, 2])

        def __init__(self, p): self.p = np.array(p)
        def predict_proba(self, X): return np.tile(self.p, (len(X), 1))

    ens = AveragedEnsemble({"a": Fixed([1, 0, 0]), "b": Fixed([0, 1, 0])}).fit()
    assert np.allclose(ens.predict_proba(pd.DataFrame({"x": [1]})), [[0.5, 0.5, 0.0]])
