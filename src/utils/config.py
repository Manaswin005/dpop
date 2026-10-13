"""Configuration loading and project-relative path resolution."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

# Project root = two levels above this file (src/utils/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load config.yaml once and cache it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path) -> Path:
    """Turn a config path (relative to project root) into an absolute Path.

    Works identically on Windows and Linux because pathlib handles separators.
    """
    return (PROJECT_ROOT / Path(relative)).resolve()


def get_logger(name: str) -> logging.Logger:
    """Simple console logger shared by all modules."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("interpret").setLevel(logging.WARNING)  # silence EBM internals
    return logging.getLogger(name)
