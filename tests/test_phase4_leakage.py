"""Phase 4 leakage checks: nothing learned from the test set."""
import inspect
import json

import numpy as np

from src.features import selection
from src.models import train as train_module
from src.preprocessing.prepare import load_split
from src.utils.config import load_config, resolve

CFG = load_config()
ART = resolve(CFG["paths"]["artifacts_dir"])


def test_cv_results_cover_training_rows_only():
    train, _ = load_split()
    for name in ["EBM", "RuleFit", "GPC", "QDA"]:
        oof = np.load(ART / "cv" / f"{name}_oof.npy")
        assert oof.shape == (len(train), 3), f"{name} CV predictions are not training-set sized"
        rec = json.loads((ART / "cv" / f"{name}.json").read_text())
        assert rec["n_train_rows"] <= len(train)


def test_feature_selection_reproduces_from_training_set_alone():
    train, _ = load_split()
    scores, assoc = selection.score_features(train)
    kept = set(selection.decide(scores, assoc).query("decision == 'KEEP'")["feature"])
    saved = json.loads((ART / "selected_features.json").read_text())
    assert kept == set(saved["categorical"]) | set(saved["numerical"])


def test_training_code_never_reads_test_labels():
    # Training, tuning and feature selection only unpack the training part.
    for fn in (train_module.train_models, train_module.tune_qda, selection.main):
        src = inspect.getsource(fn)
        assert "y_test" not in src and "X_test" not in src


def test_winner_matches_best_cv_score():
    meta = json.loads((ART / "model_metadata.json").read_text())
    cv = meta["cv_scores"]
    assert meta["selected_model"] == max(cv, key=cv.get)
    assert meta["model_status"]["TabPFN"] == "BLOCKED"
