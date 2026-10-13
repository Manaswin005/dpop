"""End-to-end execution pipeline for Stage 2 pilot data construction.

Coordinates cohort selection, cached ISR document retrieval, text parsing,
provenance tracking, and data quality reporting.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.stage2.cohort import select_pilot_cohort
from src.stage2.fetcher import fetch_project_isr_metadata, download_isr_text
from src.stage2.parser import parse_isr_document


def _parse_wb_date(d_str: str | None) -> datetime | None:
    if not d_str or pd.isna(d_str):
        return None
    # Common formats: 2015-04-01T04:00:00Z, 5/19/2009, 20-May-2008, 12/14/2018
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d-%b-%Y",
        "%d-%B-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(d_str).strip(), fmt)
        except Exception:
            continue
    return None


def run_stage2_pilot_pipeline(
    max_external_ops: int = 60,
    stop_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Stage 2 pilot data pipeline."""
    # 1. Select cohort
    cohort = select_pilot_cohort()

    snapshots = []
    provenance = []

    external_ops = 0

    for idx, row in cohort.iterrows():
        pid = row["Project ID"]
        outcome = row["outcome_class"]
        app_date_str = row.get("Approval Date")
        eval_date_str = row.get("IEG_EvalDate")
        deact_date_str = row.get("Deactivation Date")

        d_app = _parse_wb_date(app_date_str)
        d_eval = _parse_wb_date(eval_date_str)
        d_deact = _parse_wb_date(deact_date_str)

        # 2. Fetch metadata
        meta = fetch_project_isr_metadata(pid)
        if "error" in meta and not meta.get("documents"):
            provenance.append({
                "project_id": pid,
                "outcome_class": outcome,
                "status": "METADATA_FETCH_ERROR",
                "exclusion_reason": meta.get("error", "API error"),
                "doc_id": None,
                "repnb": None,
                "doc_date": None,
                "doc_url": None,
            })
            if stop_on_error:
                break
            continue

        raw_docs = meta.get("documents", {})
        doc_list = [v for k, v in raw_docs.items() if k != "facets" and isinstance(v, dict)]

        if not doc_list:
            provenance.append({
                "project_id": pid,
                "outcome_class": outcome,
                "status": "NO_ISRS_AVAILABLE",
                "exclusion_reason": "No publicly disclosed ISRs exist for project",
                "doc_id": None,
                "repnb": None,
                "doc_date": None,
                "doc_url": None,
            })
            continue

        # Sort documents chronologically by docdt
        doc_list = sorted(doc_list, key=lambda x: str(x.get("docdt", "")))

        # 3. Filter for valid interim snapshots
        eligible_docs = []
        for d in doc_list:
            d_date = _parse_wb_date(d.get("docdt"))
            title = d.get("display_title", "")
            # Exclude completion / ICR documents
            if "completion" in title.lower() or "icr" in title.lower():
                continue
            # Must strictly follow approval
            if d_app and d_date and d_date <= d_app:
                continue
            # Must strictly precede final IEG evaluation
            if d_eval and d_date and d_date >= d_eval:
                continue
            # Preferably precede deactivation
            if d_deact and d_date and d_date > d_deact:
                continue
            eligible_docs.append((d_date, d))

        if not eligible_docs:
            provenance.append({
                "project_id": pid,
                "outcome_class": outcome,
                "status": "NO_ELIGIBLE_INTERIM_SNAPSHOT",
                "exclusion_reason": "No disclosed ISRs fall strictly between approval and closure/evaluation",
                "doc_id": None,
                "repnb": None,
                "doc_date": None,
                "doc_url": None,
            })
            continue

        # 4. Select candidate interim snapshot
        # Prefer Sequence 02 if present; otherwise the earliest available interim report
        selected_doc = None
        for _, d in eligible_docs:
            title = d.get("display_title", "")
            if "Sequence 02" in title or "Seq No: 2" in title or "Seq No: 02" in title:
                selected_doc = d
                break
        if not selected_doc:
            # Choose earliest available interim
            selected_doc = eligible_docs[0][1]

        doc_id = str(selected_doc.get("id"))
        repnb = selected_doc.get("repnb")
        doc_date = selected_doc.get("docdt")
        doc_url = selected_doc.get("url")
        txt_url = selected_doc.get("txturl")

        # 5. Download text document
        text = download_isr_text(pid, doc_id, txt_url)
        if not text:
            provenance.append({
                "project_id": pid,
                "outcome_class": outcome,
                "status": "TEXT_DOWNLOAD_FAILED",
                "exclusion_reason": "Text document unavailable or connection closed",
                "doc_id": doc_id,
                "repnb": repnb,
                "doc_date": doc_date,
                "doc_url": doc_url,
            })
            continue

        # 6. Parse document
        parsed = parse_isr_document(text)

        snapshot_record = {
            "project_id": pid,
            "snapshot_doc_id": doc_id,
            "snapshot_repnb": repnb,
            "snapshot_date": doc_date[:10] if doc_date else parsed.get("archive_date"),
            "snapshot_sequence": parsed.get("sequence_number"),
            "pdo_rating_current": parsed.get("pdo_rating_current"),
            "pdo_rating_previous": parsed.get("pdo_rating_previous"),
            "ip_rating_current": parsed.get("ip_rating_current"),
            "ip_rating_previous": parsed.get("ip_rating_previous"),
            "overall_risk_rating": parsed.get("overall_risk_rating"),
            "disbursement_percentage": parsed.get("disbursement_percentage"),
            "schedule_delay_months": parsed.get("schedule_delay_months"),
            "outcome_class": outcome,  # Evaluation target only
        }
        snapshots.append(snapshot_record)

        provenance.append({
            "project_id": pid,
            "outcome_class": outcome,
            "status": "VALID_INTERIM_SNAPSHOT",
            "exclusion_reason": None,
            "doc_id": doc_id,
            "repnb": repnb,
            "doc_date": doc_date,
            "doc_url": doc_url,
        })

    df_snapshots = pd.DataFrame(snapshots)
    df_prov = pd.DataFrame(provenance)

    out_dir = Path("data/stage2")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_snapshots.to_csv(out_dir / "pilot_snapshots.csv", index=False)
    df_prov.to_csv(out_dir / "pilot_provenance.csv", index=False)

    return df_snapshots, df_prov


if __name__ == "__main__":
    snaps, prov = run_stage2_pilot_pipeline()
    print(f"Pipeline finished. Valid snapshots: {len(snaps)} / {len(prov)}")
    print(prov["status"].value_counts())
