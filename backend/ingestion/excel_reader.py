import sys
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    GOOGLE_SHEET_ID, GOOGLE_CREDS_PATH,
    SHEET_LAWS, SHEET_SECTIONS, SHEET_DEFINITIONS,
    SHEET_VERSIONS, SHEET_CROSSREFS, SHEET_CASES
)

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def _get_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(str(GOOGLE_CREDS_PATH), scopes=_SCOPES)
    return gspread.authorize(creds)


def _read_sheet(sheet_name: str) -> list:
    """Read all rows from a Google Sheet tab. Row 1 is used as headers.
    Skips columns with empty/duplicate headers to handle sheets with trailing blank columns."""
    client = _get_client()
    sh     = client.open_by_key(GOOGLE_SHEET_ID)
    ws     = sh.worksheet(sheet_name)
    all_values = ws.get_all_values()
    if not all_values:
        return []

    # Build header list, keeping only columns with valid unique headers
    raw_headers = all_values[0]
    seen        = set()
    valid_cols  = []   # list of (col_index, header_name)
    for i, h in enumerate(raw_headers):
        h = str(h).strip()
        if h and h not in seen:
            seen.add(h)
            valid_cols.append((i, h))

    out = []
    for row in all_values[1:]:
        record = {h: (str(row[i]).strip() if i < len(row) else "") for i, h in valid_cols}
        if any(v for v in record.values()):   # skip fully empty rows
            out.append(record)
    return out


def read_laws() -> list:
    raw = _read_sheet(SHEET_LAWS)
    out = []
    for r in raw:
        uid  = r.get("law_uid", "").strip()
        name = r.get("law_name", "").strip()
        if not uid or not name:
            continue
        try:
            year = int(r.get("enactment_year", "") or 0) or None
        except ValueError:
            year = None
        out.append({
            "law_uid":        uid,
            "law_name":       name,
            "domain":         r.get("domain", ""),
            "law_type":       r.get("law_type_central_state", ""),
            "jurisdiction":   r.get("jurisdiction", ""),
            "act_number":     r.get("act_number_short_title", ""),
            "enactment_year": year,
            "status":         r.get("initial_status", ""),
        })
    return out


def read_sections() -> list:
    raw = _read_sheet(SHEET_SECTIONS)
    out = []
    for r in raw:
        uid     = r.get("section_uid", "").strip()
        law_uid = r.get("law_uid", "").strip()
        if not uid or not law_uid:
            continue
        out.append({
            "section_uid":   uid,
            "law_uid":       law_uid,
            "section_label": r.get("section_label", ""),
            "keywords":      r.get("keywords", ""),
            "exact_text":    r.get("exact_text", ""),
        })
    return out


def read_definitions() -> list:
    raw = _read_sheet(SHEET_DEFINITIONS)
    out = []
    for r in raw:
        uid = r.get("definition_uid", "").strip()
        if not uid:
            continue
        out.append({
            "definition_uid":  uid,
            "term":            r.get("term", ""),
            "sub_section_uid": r.get("sub_section_uid", ""),
            "definition_text": r.get("definition_text", ""),
        })
    return out


def read_versions() -> list:
    raw = _read_sheet(SHEET_VERSIONS)
    out = []
    for r in raw:
        uid = r.get("version_uid", "").strip()
        if not uid:
            continue
        out.append({
            "version_uid":        uid,
            "section_uid":        r.get("section_uid", ""),
            "law_uid":            r.get("law_uid", ""),
            "version_type":       r.get("version_type", "").strip(),
            "effective_from":     r.get("effective_from", ""),
            "effective_to":       r.get("effective_to", ""),
            "summary_legal_text": r.get("summary_legal_text", ""),
        })
    return out


def read_crossrefs() -> list:
    raw = _read_sheet(SHEET_CROSSREFS)
    out = []
    for r in raw:
        uid = r.get("crossref_uid", "").strip()
        if not uid:
            continue
        out.append({
            "crossref_uid":      uid,
            "from_section_uid":  r.get("from_section_uid", ""),
            "from_law_uid":      r.get("from_law_uid", ""),
            "to_section_uid":    r.get("to_section_uid", ""),
            "to_law_uid":        r.get("to_law_uid", ""),
            "relationship_type": r.get("relationship_type", ""),
        })
    return out


def read_cases() -> list:
    raw = _read_sheet(SHEET_CASES)
    out = []
    for r in raw:
        uid = r.get("case_uid", "").strip()
        if not uid:
            continue
        try:
            year = int(r.get("year", "") or 0) or None
        except ValueError:
            year = None
        out.append({
            "case_uid":             uid,
            "case_name":            r.get("case_name", ""),
            "court":                r.get("court", ""),
            "citation":             r.get("citation", ""),
            "year":                 year,
            "governing_law":        r.get("governing_law", ""),
            "sections_interpreted": r.get("sections_interpreted", ""),
            "key_holding":          r.get("key_holding", ""),
            "interpretive_effect":  r.get("interpretive_effect", ""),
        })
    return out
