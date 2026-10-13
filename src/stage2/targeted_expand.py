"""Targeted Phase 6 expansion for 3 Unsuccessful projects: P107843, P126576, P126791.

Strictly bounded to maximum 3 metadata + 3 document calls.
Local cache checked first. Stops immediately on any timeout or blocker.
"""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request
import pandas as pd

from src.stage2.fetcher import download_isr_text
from src.stage2.parser import parse_isr_document
from src.stage2.pipeline import _parse_wb_date

TARGET_PIDS = ["P107843", "P126576", "P126791"]


def run_targeted_expansion() -> None:
    cohort = pd.read_csv("data/stage2/pilot_cohort.csv").set_index("Project ID")
    snaps_path = Path("data/stage2/pilot_snapshots.csv")
    prov_path = Path("data/stage2/pilot_provenance.csv")

    snaps = pd.read_csv(snaps_path)
    prov = pd.read_csv(prov_path)

    meta_cache_dir = Path("data/stage2/cache/metadata")
    doc_cache_dir = Path("data/stage2/cache/documents")
    meta_cache_dir.mkdir(parents=True, exist_ok=True)
    doc_cache_dir.mkdir(parents=True, exist_ok=True)

    ops_made = 0
    new_snaps = []
    new_prov = []

    print(f"Starting targeted expansion for: {TARGET_PIDS}")

    for pid in TARGET_PIDS:
        row = cohort.loc[pid]
        outcome = row["outcome_class"]
        app_date_str = row["Approval Date"]
        eval_date_str = row["IEG_EvalDate"]
        deact_date_str = row["Deactivation Date"]

        d_app = _parse_wb_date(app_date_str)
        d_eval = _parse_wb_date(eval_date_str)
        d_deact = _parse_wb_date(deact_date_str)

        # 1. Fetch metadata (cache first)
        cache_meta_file = meta_cache_dir / f"{pid}.json"
        meta = None
        if cache_meta_file.exists():
            try:
                meta = json.loads(cache_meta_file.read_text(encoding="utf-8"))
                print(f"{pid}: Metadata found in local cache.")
            except Exception:
                pass

        if meta is None:
            ops_made += 1
            print(f"{pid}: External call #{ops_made} -> Fetching metadata from WDS API...")
            url = (
                f"https://search.worldbank.org/api/v3/wds?format=json&qterm={pid}"
                f"&docty=Implementation%20Status%20and%20Results%20Report&rows=30"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    meta = json.loads(resp.read().decode("utf-8"))
                    cache_meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"{pid}: Metadata fetch FAILED/TIMEOUT -> {e}")
                new_prov.append({
                    "project_id": pid,
                    "outcome_class": outcome,
                    "status": "METADATA_FETCH_ERROR",
                    "exclusion_reason": str(e),
                    "doc_id": None, "repnb": None, "doc_date": None, "doc_url": None,
                })
                print("Stopping immediately per protocol.")
                break

        raw_docs = meta.get("documents", {})
        doc_list = [v for k, v in raw_docs.items() if k != "facets" and isinstance(v, dict)]

        if not doc_list:
            print(f"{pid}: 0 ISRs found in WDS metadata.")
            new_prov.append({
                "project_id": pid,
                "outcome_class": outcome,
                "status": "NO_ISRS_AVAILABLE",
                "exclusion_reason": "No disclosed ISRs exist",
                "doc_id": None, "repnb": None, "doc_date": None, "doc_url": None,
            })
            continue

        doc_list = sorted(doc_list, key=lambda x: str(x.get("docdt", "")))

        # Strict temporal filtering
        eligible_docs = []
        for d in doc_list:
            d_date = _parse_wb_date(d.get("docdt"))
            title = d.get("display_title", "")
            if "completion" in title.lower() or "icr" in title.lower():
                continue
            if d_app and d_date and d_date <= d_app:
                continue
            if d_eval and d_date and d_date >= d_eval:
                continue
            if d_deact and d_date and d_date > d_deact:
                continue
            eligible_docs.append((d_date, d))

        if not eligible_docs:
            print(f"{pid}: No eligible interim snapshot satisfying strict dates.")
            new_prov.append({
                "project_id": pid,
                "outcome_class": outcome,
                "status": "NO_ELIGIBLE_INTERIM_SNAPSHOT",
                "exclusion_reason": "No disclosed ISRs fall strictly between approval and closure/eval",
                "doc_id": None, "repnb": None, "doc_date": None, "doc_url": None,
            })
            continue

        # Choose Sequence 02 if present, else earliest
        selected_doc = None
        for _, d in eligible_docs:
            t = d.get("display_title", "")
            if "Sequence 02" in t or "Seq No: 2" in t or "Seq No: 02" in t:
                selected_doc = d
                break
        if not selected_doc:
            selected_doc = eligible_docs[0][1]

        doc_id = str(selected_doc.get("id"))
        repnb = selected_doc.get("repnb")
        doc_date = selected_doc.get("docdt")
        doc_url = selected_doc.get("url")
        txt_url = selected_doc.get("txturl")

        # 2. Download text (cache first)
        cache_doc_file = doc_cache_dir / f"{pid}_{doc_id}.txt"
        text = None
        if cache_doc_file.exists():
            text = cache_doc_file.read_text(encoding="utf-8", errors="ignore")
            print(f"{pid}: Document text found in local cache.")
        else:
            ops_made += 1
            print(f"{pid}: External call #{ops_made} -> Downloading document text {doc_id}...")
            text = download_isr_text(pid, doc_id, txt_url)

        if not text:
            print(f"{pid}: Text download FAILED.")
            new_prov.append({
                "project_id": pid,
                "outcome_class": outcome,
                "status": "TEXT_DOWNLOAD_FAILED",
                "exclusion_reason": "Text document download failed or connection closed",
                "doc_id": doc_id, "repnb": repnb, "doc_date": doc_date, "doc_url": doc_url,
            })
            continue

        # 3. Parse text
        parsed = parse_isr_document(text)
        print(f"{pid}: SUCCESS! Parsed snapshot {doc_date[:10]} - PDO: {parsed.get('pdo_rating_current')}, IP: {parsed.get('ip_rating_current')}, Disb%: {parsed.get('disbursement_percentage')}, Delay: {parsed.get('schedule_delay_months')}")

        snap_rec = {
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
            "outcome_class": outcome,
        }
        new_snaps.append(snap_rec)
        new_prov.append({
            "project_id": pid,
            "outcome_class": outcome,
            "status": "VALID_INTERIM_SNAPSHOT",
            "exclusion_reason": None,
            "doc_id": doc_id, "repnb": repnb, "doc_date": doc_date, "doc_url": doc_url,
        })

    print(f"\n=== TARGETED EXPANSION SUMMARY ===")
    print(f"External calls made: {ops_made}")
    print(f"New valid snapshots extracted: {len(new_snaps)}")

    if new_snaps:
        updated_snaps = pd.concat([snaps[~snaps["project_id"].isin(TARGET_PIDS)], pd.DataFrame(new_snaps)]).reset_index(drop=True)
        updated_snaps.to_csv(snaps_path, index=False)
        print(f"Updated pilot_snapshots.csv -> Total rows now: {len(updated_snaps)}")
        print("Updated Outcome Counts:")
        print(updated_snaps["outcome_class"].value_counts())

    if new_prov:
        updated_prov = pd.concat([prov[~prov["project_id"].isin(TARGET_PIDS)], pd.DataFrame(new_prov)]).reset_index(drop=True)
        updated_prov.to_csv(prov_path, index=False)
        print(f"Updated pilot_provenance.csv -> Total rows now: {len(updated_prov)}")


if __name__ == "__main__":
    run_targeted_expansion()
