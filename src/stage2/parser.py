"""Text parser for World Bank ISR documents.

Extracts standardized supervision signals from plain-text disclosable ISRs.
Enforces strict null handling: missing or ambiguous values return NaN.
"""
from __future__ import annotations

from datetime import datetime
import re
import numpy as np


VALID_RATINGS = [
    "Highly Satisfactory",
    "Moderately Satisfactory",
    "Moderately Unsatisfactory",
    "Satisfactory",
    "Highly Unsatisfactory",
    "Unsatisfactory",
]

VALID_RISKS = ["Low", "Moderate", "Substantial", "High"]


def parse_ratings_line(section_name: str, text: str) -> tuple[str | None, str | None]:
    """Extract previous and current ratings from a section line."""
    m = re.search(re.escape(section_name) + r"[^\n]*", text, re.IGNORECASE)
    if not m:
        return None, None

    line = m.group(0)[len(section_name):]
    matches = []
    for r in VALID_RATINGS:
        for hit in re.finditer(re.escape(r), line, re.IGNORECASE):
            matches.append((hit.start(), r))
    matches.sort()

    # Deduplicate overlapping matches (e.g. Moderately Satisfactory vs Satisfactory)
    filtered = []
    last_end = -1
    for start, r in matches:
        if start >= last_end:
            filtered.append(r)
            last_end = start + len(r)

    if len(filtered) >= 2:
        return filtered[0], filtered[1]
    elif len(filtered) == 1:
        return None, filtered[0]
    return None, None


def parse_dates_for_delay(text: str) -> float:
    """Parse original and revised closing dates from the report.
    
    Returns delay in months as float, 0.0 if identical, or NaN if unverified/missing.
    """
    orig_str = None
    rev_str = None

    # Pattern 1: Key Dates section
    # e.g., Original Closing Date:31-Dec-2013   Revised Closing Date:30-Jun-2015
    m1 = re.search(
        r"Original Closing Date\s*[:\s]*(\d{1,2}-[A-Za-z]{3}-\d{4})", text, re.IGNORECASE
    )
    m2 = re.search(
        r"Revised Closing Date\s*[:\s]*(\d{1,2}-[A-Za-z]{3}-\d{4})", text, re.IGNORECASE
    )
    if m1 and m2:
        orig_str = m1.group(1)
        rev_str = m2.group(1)
    else:
        # Pattern 2: Financial Agreement(s) Key Dates table
        m_table = re.search(
            r"(\d{1,2}-[A-Za-z]{3}-\d{4})\s+(\d{1,2}-[A-Za-z]{3}-\d{4})\s*(?:\n|Public)",
            text,
        )
        if m_table:
            orig_str = m_table.group(1)
            rev_str = m_table.group(2)

    if not orig_str or not rev_str:
        return np.nan

    try:
        d_orig = datetime.strptime(orig_str, "%d-%b-%Y")
        d_rev = datetime.strptime(rev_str, "%d-%b-%Y")
        diff_months = (d_rev.year - d_orig.year) * 12 + (d_rev.month - d_orig.month)
        return float(max(0.0, diff_months))
    except Exception:
        return np.nan


def parse_disbursement_percentage(text: str) -> float:
    """Extract % Disbursed from financial performance section."""
    m = re.search(r"%\s*Disbursed\s*[:\s]+(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Match in table row ending with % disbursed number
    m_row = re.search(
        r"(?:USD|XDR|EUR)\s+[\d\.,]+\s+[\d\.,]+\s+[\d\.,]+\s+[\d\.,]+\s+[\d\.,]+\s+(\d+(?:\.\d+)?)",
        text,
    )
    if m_row:
        try:
            val = float(m_row.group(1))
            if 0.0 <= val <= 200.0:
                return val
        except ValueError:
            pass

    return np.nan


def parse_isr_document(text: str) -> dict:
    """Parse all target implementation fields from a single ISR text."""
    data = {
        "sequence_number": np.nan,
        "archive_date": None,
        "pdo_rating_current": None,
        "pdo_rating_previous": None,
        "ip_rating_current": None,
        "ip_rating_previous": None,
        "overall_risk_rating": None,
        "disbursement_percentage": np.nan,
        "schedule_delay_months": np.nan,
    }

    if not text:
        return data

    # 1. Sequence Number
    m_seq = re.search(
        r"Seq(?:uence|\.)?\s*(?:No)?\.?\s*[:\s]*(\d+)", text, re.IGNORECASE
    )
    if m_seq:
        data["sequence_number"] = int(m_seq.group(1))

    # 2. Archive Date
    m_arch = re.search(
        r"ARCHIVED\s+on\s+(\d{1,2}-[A-Za-z]{3}-\d{4})", text, re.IGNORECASE
    )
    if m_arch:
        data["archive_date"] = m_arch.group(1)

    # 3. PDO Rating
    prev_pdo, curr_pdo = parse_ratings_line("Progress towards achievement of PDO", text)
    if not curr_pdo:
        prev_pdo, curr_pdo = parse_ratings_line("Progress towards achievement of GEO", text)
    data["pdo_rating_previous"] = prev_pdo
    data["pdo_rating_current"] = curr_pdo

    # 4. IP Rating
    prev_ip, curr_ip = parse_ratings_line("Overall Implementation Progress (IP)", text)
    data["ip_rating_previous"] = prev_ip
    data["ip_rating_current"] = curr_ip

    # 5. Overall Risk Rating
    m_risk = re.search(r"Overall Risk Rating[^\n]*", text, re.IGNORECASE)
    if m_risk:
        line = m_risk.group(0)
        found = [r for r in VALID_RISKS if r.lower() in line.lower()]
        if found:
            data["overall_risk_rating"] = found[-1]

    # 6. Schedule delay
    data["schedule_delay_months"] = parse_dates_for_delay(text)

    # 7. Disbursement percentage
    data["disbursement_percentage"] = parse_disbursement_percentage(text)

    return data
