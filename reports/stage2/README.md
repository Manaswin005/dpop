# Stage 2 Data Pilot — Implementation-Stage Supervision

This module implements the isolated data construction pilot for **Stage 2: Implementation-Stage Risk Prediction** using official World Bank Implementation Status and Results Reports (ISRs).

It operates completely independently of the Stage 1 approval-time system and never alters Stage 1 pipelines, models, or data.

---

## 1. Directory Structure

- `src/stage2/cohort.py`: Reproducible cohort selection from `train.csv` (IPF, FY $\ge$ 2008, random seed 42).
- `src/stage2/fetcher.py`: Bounded retrieval from official World Bank WDS API and disclosable text endpoints, with strict local file caching.
- `src/stage2/parser.py`: Extraction of 6 standardized supervision signals from plain-text ISRs with explicit NaN semantics.
- `src/stage2/pipeline.py`: Orchestrator generating snapshots and full provenance records.
- `data/stage2/cache/`: Local two-tier cache of raw JSON metadata and document texts.
- `data/stage2/pilot_snapshots.csv`: Validated interim snapshots.
- `data/stage2/pilot_provenance.csv`: Traceability log with official document IDs, URLs, archive dates, and exclusion reasons.
- `tests/test_stage2_pilot.py`: Pytest suite validating date ordering, no-leakage, and missing-value handling.

---

## 2. Running Locally (No-Network / Cache-First)

To re-run the pilot pipeline using the existing local cache (zero network calls):

```powershell
# From the project root:
.\.venv\Scripts\Activate.ps1

# Run the pipeline (reads cached files directly):
python -m src.stage2.pipeline

# Run the Stage 2 tests:
pytest tests/test_stage2_pilot.py -v
```

---

## 3. External Network Policy

If refreshing or expanding the pilot:
- External requests are restricted exclusively to:
  - `https://search.worldbank.org/api/v3/wds`
  - `https://documents1.worldbank.org/`
- Every successful call is permanently cached in `data/stage2/cache/` to prevent redundant requests.
- Hard bound: $\le 60$ total external operations per execution.
- If an external call encounters an error or timeout, the pipeline stops immediately and preserves all work completed up to that point.
