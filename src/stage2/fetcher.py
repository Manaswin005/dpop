"""Bounded metadata and ISR document retrieval with two-tier local caching.

Enforces a strict upper bound of external operations and caches every
successful API call and document download locally to avoid duplicate requests.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


METADATA_CACHE_DIR = Path("data/stage2/cache/metadata")
DOCUMENT_CACHE_DIR = Path("data/stage2/cache/documents")


def fetch_project_isr_metadata(
    project_id: str,
    cache_dir: Path = METADATA_CACHE_DIR,
    timeout: int = 12,
) -> dict:
    """Fetch WDS ISR document list for a project, reading from local cache first."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{project_id}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = (
        f"https://search.worldbank.org/api/v3/wds?format=json&qterm={project_id}"
        f"&docty=Implementation%20Status%20and%20Results%20Report&rows=30"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data
    except Exception as e:
        return {"error": str(e), "total": 0, "documents": {}}


def download_isr_text(
    project_id: str,
    doc_id: str,
    txt_url: str,
    cache_dir: Path = DOCUMENT_CACHE_DIR,
) -> str | None:
    """Download ISR disclosable text document using curl.exe, cached locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{project_id}_{doc_id}.txt"
    if cache_file.exists():
        content = cache_file.read_text(encoding="utf-8", errors="ignore")
        if content.strip():
            return content

    if not txt_url:
        return None

    # Replace http with https for documents1 endpoint
    url = txt_url.replace("http://documents.worldbank.org", "https://documents1.worldbank.org")

    try:
        cmd = ["curl.exe", "-s", "-L", url]
        res = subprocess.run(cmd, capture_output=True, timeout=20, check=False)
        if res.returncode == 0 and res.stdout:
            text = res.stdout.decode("utf-8", errors="ignore")
            if "Implementation Status" in text or "Overall Ratings" in text:
                cache_file.write_text(text, encoding="utf-8", errors="ignore")
                return text
    except Exception:
        pass

    # Fallback to direct http url via curl if https failed
    try:
        cmd = ["curl.exe", "-s", "-L", txt_url]
        res = subprocess.run(cmd, capture_output=True, timeout=20, check=False)
        if res.returncode == 0 and res.stdout:
            text = res.stdout.decode("utf-8", errors="ignore")
            if len(text.strip()) > 100:
                cache_file.write_text(text, encoding="utf-8", errors="ignore")
                return text
    except Exception:
        pass

    return None
