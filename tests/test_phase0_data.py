"""Phase 0 tests: schema, target mapping, record selection, eligibility, leakage guards."""
import pandas as pd
import pytest

from src.data import cleaning
from src.data.feature_eligibility import eligibility_table, included_features
from src.data.load import EXPECTED_COLUMNS, SchemaError, load_raw, validate_schema
from src.utils.config import load_config

CFG = load_config()

# Columns that must never be predictors (post-approval or target)
FORBIDDEN = {
    "IEG_Outcome", "ERR at Completion", "Deactivation Date", "Exit FY", "Net Commitment",
    "IEG_EvalDate", "IEG_EvalFY", "IEG_EvalType", "IEG_SourceDocumentURL", "Project ID", "Project Name",
}


@pytest.fixture(scope="module")
def raw():
    return load_raw()


# ---- schema ---------------------------------------------------------------
def test_schema_matches(raw):
    assert raw.shape == (12932, 35)
    validate_schema(raw)


def test_schema_error_on_missing_column(raw):
    with pytest.raises(SchemaError):
        validate_schema(raw.drop(columns=["IEG_Outcome"]))


# ---- target mapping -------------------------------------------------------
def test_target_mapping_counts(raw):
    mapped, summary = cleaning.map_target(raw)
    assert summary["excluded_counts"] == {"Not Rated": 162, "Not Available": 9, "Not Applicable": 9}
    assert len(mapped) == 12752
    assert set(mapped[CFG["columns"]["target"]]) == set(CFG["target"]["class_order"])


def test_target_mapping_rules():
    df = pd.DataFrame({"IEG_Outcome": ["Highly Satisfactory", "Moderately Unsatisfactory",
                                       "Highly Unsatisfactory", "Not Rated"]})
    mapped, _ = cleaning.map_target(df)
    assert mapped["outcome_class"].tolist() == ["Successful", "Partially Successful", "Unsuccessful"]


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        cleaning.map_target(pd.DataFrame({"IEG_Outcome": ["Excellent"]}))


# ---- record selection -----------------------------------------------------
def test_one_record_per_project(raw):
    mapped, _ = cleaning.map_target(raw)
    selected, _ = cleaning.select_one_evaluation_per_project(mapped)
    assert selected["Project ID"].is_unique
    assert len(selected) == mapped["Project ID"].nunique()


def test_latest_evaluation_kept_and_tie_break():
    df = pd.DataFrame({
        "Project ID": ["P1", "P1", "P2", "P2"],
        "IEG_EvalDate": ["1/1/2000", "43447", "5/5/2010", "5/5/2010"],  # 43447 = 2018-12-13
        "IEG_EvalType": ["ES", "ES", "ES", "PAR"],
        "outcome_class": ["Successful", "Unsuccessful", "Successful", "Unsuccessful"],
    })
    sel, _ = cleaning.select_one_evaluation_per_project(df)
    got = sel.set_index("Project ID")["outcome_class"].to_dict()
    assert got == {"P1": "Unsuccessful", "P2": "Unsuccessful"}


def test_mixed_date_parsing():
    parsed = cleaning.parse_mixed_date(pd.Series(["6/29/2018", "43447"]))
    assert parsed.tolist() == [pd.Timestamp("2018-06-29"), pd.Timestamp("2018-12-13")]


# ---- data-quality fixes ---------------------------------------------------
def test_upstream_fill_converted_to_missing():
    df = pd.DataFrame({"Lending Project Cost": [55700000.0, 1e6]})
    out = cleaning.flag_upstream_imputation(df)
    assert out["Lending Project Cost"].isna().tolist() == [True, False]
    assert out["lending_cost_reported"].tolist() == [0, 1]


def test_approval_fy_repair_and_region():
    df = pd.DataFrame({"Approval FY": [0.0, 2001.0], "Approval Date": ["12/15/2005", "3/1/2001"],
                       "Region": ["AFE", "EAP"]})
    fixed, info = cleaning.fix_approval_fy(df)
    assert fixed["Approval FY"].tolist() == [2006, 2001] and info["invalid_replaced"] == 1
    assert cleaning.harmonise_region(df)["Region"].tolist() == ["AFR", "EAP"]


# ---- eligibility / leakage guards -----------------------------------------
def test_every_column_has_a_decision():
    assert sorted(eligibility_table()["column"]) == sorted(EXPECTED_COLUMNS)


def test_no_forbidden_or_ieg_rating_feature():
    feats = set(included_features())
    assert not feats & FORBIDDEN
    assert not any(f.startswith(("IEG_", "(disc)")) for f in feats)


def test_config_features_match_register():
    cfg_feats = set(CFG["features"]["categorical"]) | set(CFG["features"]["numerical"])
    assert cfg_feats == set(included_features())
