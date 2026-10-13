"""Lifecycle Project Store (Stage 1 Approval + Stage 2 Monitoring Records).

Persists project metadata, initial approval-time predictions (Stage 1),
and dated implementation-stage monitoring records (Stage 2) under a unique project ID.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

from src.utils.config import resolve

DEFAULT_STORAGE_FILE = "data/stage2/lifecycle_projects.json"


def _get_storage_path(custom_path: str | Path | None = None) -> Path:
    if custom_path is not None:
        p = Path(custom_path)
    else:
        p = resolve(DEFAULT_STORAGE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_all_projects(storage_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = _get_storage_path(storage_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_all_projects(projects: dict[str, dict[str, Any]], storage_path: str | Path | None = None) -> None:
    path = _get_storage_path(storage_path)
    path.write_text(json.dumps(projects, indent=2), encoding="utf-8")


def get_project(project_id: str, storage_path: str | Path | None = None) -> dict[str, Any] | None:
    projects = load_all_projects(storage_path)
    return projects.get(project_id)


def create_or_update_project(
    project_id: str | None,
    title: str,
    description: str,
    stage1_inputs: dict[str, Any] | None = None,
    stage1_prediction: dict[str, Any] | None = None,
    created_date: str | None = None,
    storage_path: str | Path | None = None,
) -> str:
    """Create a new project or update approval-time metadata."""
    projects = load_all_projects(storage_path)

    if not project_id:
        # Generate an 8-character unique project ID
        project_id = f"PRJ-{uuid.uuid4().hex[:8].upper()}"

    existing = projects.get(project_id, {})

    project_record = {
        "project_id": project_id,
        "title": title.strip() if title else existing.get("title", "Untitled Project"),
        "description": description.strip() if description else existing.get("description", ""),
        "created_date": created_date or existing.get("created_date"),
        "stage1_inputs": stage1_inputs if stage1_inputs is not None else existing.get("stage1_inputs"),
        "stage1_prediction": stage1_prediction if stage1_prediction is not None else existing.get("stage1_prediction"),
        "monitoring_records": existing.get("monitoring_records", []),
    }

    projects[project_id] = project_record
    save_all_projects(projects, storage_path)
    return project_id


def add_monitoring_record(
    project_id: str,
    record_date: str,
    indicators: dict[str, Any],
    interim_assessment: dict[str, Any] | None = None,
    notes: str = "",
    storage_path: str | Path | None = None,
) -> bool:
    """Add a dated implementation-stage monitoring record."""
    projects = load_all_projects(storage_path)
    if project_id not in projects:
        return False

    record = {
        "record_id": f"REC-{uuid.uuid4().hex[:6].upper()}",
        "record_date": str(record_date),
        "indicators": indicators,
        "interim_assessment": interim_assessment or {},
        "notes": notes,
    }

    projects[project_id]["monitoring_records"].append(record)
    save_all_projects(projects, storage_path)
    return True
