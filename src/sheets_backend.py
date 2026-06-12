# src/sheets_backend.py
# ─────────────────────────────────────────────────────────────────────────────
# Google Sheets Backend — persistent storage for crowdsourced essay submissions
#
# Why: Streamlit Cloud's filesystem is temporary. Writing to essays.csv directly
# means submissions can vanish when the app restarts/redeploys. Google Sheets
# gives you a free, permanent, live-viewable database with zero infrastructure.
#
# Falls back to local CSV automatically if Sheets credentials aren't configured
# — so this works identically when developing locally without any setup.
# ─────────────────────────────────────────────────────────────────────────────

import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from config import RAW_ESSAYS_DIR

ESSAYS_CSV = RAW_ESSAYS_DIR / "essays.csv"
CSV_HEADERS = ["question", "essay", "mark", "max_marks", "level", "feedback", "topic", "date_added", "source"]

# Worksheet tab name used inside the Google Sheet
WORKSHEET_NAME = "essays"


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _get_streamlit_secrets():
    """
    Safely tries to read st.secrets without crashing if Streamlit
    isn't running or secrets.toml doesn't exist.
    """
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets and "sheet_url" in st.secrets:
            return st.secrets
    except Exception:
        pass
    return None


def is_sheets_configured() -> bool:
    """True if Google Sheets credentials are available."""
    return _get_streamlit_secrets() is not None


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _get_worksheet():
    """
    Connects to Google Sheets using the service account credentials
    stored in st.secrets. Returns the worksheet object, or None on failure.
    """
    secrets = _get_streamlit_secrets()
    if secrets is None:
        return None

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds_dict = dict(secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        sheet = client.open_by_url(secrets["sheet_url"])

        # Get or create the worksheet tab
        try:
            worksheet = sheet.worksheet(WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=WORKSHEET_NAME, rows=1000, cols=len(CSV_HEADERS))
            worksheet.append_row(CSV_HEADERS)

        # Ensure header row exists
        first_row = worksheet.row_values(1)
        if first_row != CSV_HEADERS:
            if not first_row:
                worksheet.append_row(CSV_HEADERS)

        return worksheet

    except Exception as e:
        print(f"[sheets_backend] Could not connect to Google Sheets: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — these are the functions streamlit_app.py calls
# Same interface regardless of which backend is active
# ─────────────────────────────────────────────────────────────────────────────

def save_essay(data: dict) -> tuple[bool, str]:
    """
    Saves one essay submission.
    Returns (success, backend_used) where backend_used is 'sheets' or 'csv'.
    """
    # Ensure all expected fields are present and in order
    row = {h: data.get(h, "") for h in CSV_HEADERS}

    worksheet = _get_worksheet()
    if worksheet is not None:
        try:
            worksheet.append_row([str(row[h]) for h in CSV_HEADERS])
            return True, "sheets"
        except Exception as e:
            print(f"[sheets_backend] Sheets write failed, falling back to CSV: {e}")

    # Fallback: local CSV
    _save_to_csv(row)
    return True, "csv"


def _save_to_csv(row: dict):
    ESSAYS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = ESSAYS_CSV.exists()
    with open(ESSAYS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_all_essays() -> pd.DataFrame:
    """
    Loads all essay submissions as a DataFrame, regardless of backend.
    Used by the dataset explorer page and by data_prep.py.
    """
    worksheet = _get_worksheet()
    if worksheet is not None:
        try:
            records = worksheet.get_all_records()
            if records:
                return pd.DataFrame(records)
            return pd.DataFrame(columns=CSV_HEADERS)
        except Exception as e:
            print(f"[sheets_backend] Sheets read failed, falling back to CSV: {e}")

    if ESSAYS_CSV.exists():
        return pd.read_csv(ESSAYS_CSV)
    return pd.DataFrame(columns=CSV_HEADERS)


def count_essays() -> int:
    """Returns the total number of submitted essays."""
    df = load_all_essays()
    return len(df)


def sync_sheets_to_csv() -> int:
    """
    Pulls everything from Google Sheets and writes it to the local
    essays.csv — used before running data_prep.py / train.py so the
    training pipeline always works on a local file regardless of backend.

    Returns the number of essays synced.
    """
    df = load_all_essays()
    if df.empty:
        return 0

    ESSAYS_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Ensure correct column order/presence
    for col in CSV_HEADERS:
        if col not in df.columns:
            df[col] = ""
    df = df[CSV_HEADERS]
    df.to_csv(ESSAYS_CSV, index=False, encoding="utf-8")
    return len(df)


def get_backend_status() -> dict:
    """Returns info about which backend is active — for display in the UI."""
    configured = is_sheets_configured()
    connected  = False
    error      = None

    if configured:
        ws = _get_worksheet()
        connected = ws is not None
        if not connected:
            error = "Credentials found but connection failed — check sheet sharing settings."

    return {
        "sheets_configured": configured,
        "sheets_connected":  connected,
        "error":             error,
        "active_backend":    "Google Sheets" if connected else "Local CSV",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run this file directly to sync Sheets -> CSV manually
# Usage: python src/sheets_backend.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    status = get_backend_status()
    print(f"Sheets configured: {status['sheets_configured']}")
    print(f"Sheets connected:  {status['sheets_connected']}")
    if status["error"]:
        print(f"Error: {status['error']}")
    print(f"Active backend:    {status['active_backend']}")
    print()

    if status["sheets_connected"]:
        n = sync_sheets_to_csv()
        print(f"✓ Synced {n} essays from Google Sheets to {ESSAYS_CSV}")
    else:
        n = count_essays()
        print(f"Using local CSV — {n} essays found at {ESSAYS_CSV}")