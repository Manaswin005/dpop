"""Phase 8B: `evaluate` must handle a missing model file without crashing.

Runs evaluate() against a temporary COPY of the artifacts, so the real
reports and model files are never touched.
"""
import json
import shutil

import pandas as pd
import pytest

from src.models import train as train_module
from src.utils.config import load_config, resolve

CFG = load_config()
REAL_ART = resolve(CFG["paths"]["artifacts_dir"])


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Copy artifacts (without GPC.joblib) to tmp and point train.py's paths there."""
    art = tmp_path / "artifacts"
    shutil.copytree(REAL_ART / "cv", art / "cv")
    (art / "models").mkdir(parents=True)
    for f in (REAL_ART / "models").glob("*.joblib"):
        if f.name != "GPC.joblib":
            shutil.copy(f, art / "models" / f.name)
    shutil.copy(REAL_ART / "selected_features.json", art / "selected_features.json")
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    monkeypatch.setattr(train_module, "resolve", lambda rel: tmp_path / rel)
    return tmp_path


def test_missing_model_is_detected(tmp_path):
    (tmp_path / "EBM.joblib").touch()
    records = {"EBM": {"status": "TRAINED"}, "GPC": {"status": "TRAINED"}, "TabPFN": {"status": "BLOCKED"}}
    missing = train_module.find_missing_artifacts(records, tmp_path)
    assert list(missing) == ["GPC"]
    msg = train_module.missing_artifact_message("GPC", missing["GPC"])
    assert "GPC.joblib" in msg and "train --models GPC" in msg


def test_evaluate_skips_missing_model_and_keeps_others(sandbox):
    real_csv = resolve(CFG["paths"]["reports_dir"]) / "model_comparison.csv"
    before = real_csv.stat().st_mtime

    train_module.evaluate()

    table = pd.read_csv(sandbox / "reports" / "model_comparison.csv")
    assert "GPC" not in set(table["model"])
    assert {"EBM", "RuleFit", "QDA"} <= set(table["model"])
    md = (sandbox / "reports" / "model_comparison.md").read_text()
    assert "GPC.joblib" in md and "train --models GPC" in md
    meta = json.loads((sandbox / "artifacts" / "model_metadata.json").read_text())
    assert meta["selected_model"] == "EBM"            # selection rule unchanged
    assert meta["missing_model_files"] == ["GPC"]
    # Results for available models match the original Phase 4 numbers
    real = pd.read_csv(real_csv).set_index("model")
    for m in ("EBM", "RuleFit", "QDA"):
        assert table.set_index("model").loc[m, "test_macro_f1"] == real.loc[m, "test_macro_f1"]
    assert real_csv.stat().st_mtime == before        # real reports untouched


def test_evaluate_stops_if_selected_model_missing(sandbox):
    (sandbox / "artifacts" / "models" / "EBM.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="CV-selected model EBM"):
        train_module.evaluate()
    assert not (sandbox / "reports" / "model_comparison.csv").exists()   # nothing written
