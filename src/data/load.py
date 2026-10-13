"""Load the raw dataset and validate that its schema matches expectations."""
from __future__ import annotations

import pandas as pd

from src.utils.config import get_logger, load_config, resolve

log = get_logger(__name__)

# The 35 columns present in the supplied file. A missing column stops the
# pipeline early with a clear message instead of failing later.
EXPECTED_COLUMNS = [
    "Project ID", "Project Name", "Region", "Country Code", "Country Name",
    "Approval Date", "Approval FY", "Sector Board", "Agreement Type",
    "Lending Project Cost", "Net Commitment", "Deactivation Date", "Exit FY",
    "Lending Instrument Type", "Lending Instrument", "Product Line Code",
    "Product Line", "IEG_EvalDate", "IEG_EvalFY", "IEG_EvalType",
    "ERR at Appraisal", "ERR at Completion", "IEG_Outcome",
    "(disc)IEG_IDImpact", "IEG_BankQualityAtEntry",
    "IEG_BankQualityOfSupervision", "IEG_OverallBankPerf", "(disc)IEG_BorrPrep",
    "IEG_ImplementingAgencyPerf", "IEG_GovernmentPerf", "IEG_OverallBorrPerf",
    "IEG_ICRQuality", "(disc)IEG_Sustainability", "IEG_MEQuality",
    "IEG_SourceDocumentURL",
]


class SchemaError(ValueError):
    """Raised when the dataset does not have the expected columns."""


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Dataset is missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        log.warning("Dataset has unexpected extra columns (ignored): %s", extra)


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV (never modified on disk) and validate its schema."""
    cfg = load_config()
    path = resolve(path or cfg["paths"]["raw_data"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV in data/raw/ "
            "or update paths.raw_data in config.yaml."
        )
    df = pd.read_csv(path)
    validate_schema(df)
    log.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df
