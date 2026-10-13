"""Phase 5 tests: SHAP output shape, additivity, and approval-time features only."""
import joblib
import numpy as np
import pandas as pd

from src.explainability.shap_explain import explain
from src.models.train import load_training_data
from src.utils.config import load_config, resolve

CFG = load_config()


def test_shap_shape_and_additivity():
    X, _, X_test, _, cats, _, order = load_training_data()
    model = joblib.load(resolve(CFG["paths"]["artifacts_dir"]) / "models" / "EBM.joblib")
    rows = X_test.head(5)
    sv = explain(model, X.sample(30, random_state=0), rows, cats, frames=[X, X_test])
    assert sv.values.shape == (5, X.shape[1], len(order))
    # base value + contributions = model probability, for every class
    assert np.allclose(sv.base_values + sv.values.sum(axis=1), model.predict_proba(rows), atol=1e-8)


def test_explained_features_are_approval_time_only():
    imp = pd.read_csv(resolve(CFG["paths"]["reports_dir"]) / "shap_global_importance.csv", index_col=0)
    forbidden = {"Net Commitment", "ERR at Appraisal", "ERR at Completion", "Exit FY", "Deactivation Date"}
    assert not set(imp.index) & forbidden
    assert not any(f.startswith(("IEG_", "(disc)")) for f in imp.index)
    _, _, _, _, cats, nums, _ = load_training_data()
    assert set(imp.index) == set(cats) | set(nums)
