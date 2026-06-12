# scripts/migrate_to_sheets.py
# ─────────────────────────────────────────────────────────────────────────────
# One-time migration: pushes essays from local data/raw_essays/essays.csv
# into your connected Google Sheet.
#
# Use this if you added essays BEFORE setting up the Google Sheets backend
# and want to consolidate everything into the Sheet.
#
# Run: python scripts/migrate_to_sheets.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
import pandas as pd
from rich.console import Console

sys.path.append(str(Path(__file__).parent.parent / "src"))
import sheets_backend as backend
from config import RAW_ESSAYS_DIR

console = Console()
ESSAYS_CSV = RAW_ESSAYS_DIR / "essays.csv"


def main():
    console.print("\n[bold cyan]Migrate Local Essays → Google Sheets[/bold cyan]\n")

    # 1. Check Sheets is connected
    status = backend.get_backend_status()
    if not status["sheets_connected"]:
        console.print("[red]✗ Google Sheets is not connected.[/red]")
        console.print(f"[yellow]  {status.get('error', 'Check your secrets.toml configuration.')}[/yellow]")
        console.print("[yellow]  Run python src/sheets_backend.py to debug the connection.[/yellow]")
        return

    console.print("[green]✓ Connected to Google Sheets[/green]")

    # 2. Load local CSV
    if not ESSAYS_CSV.exists():
        console.print(f"[red]✗ No local file found at {ESSAYS_CSV}[/red]")
        return

    local_df = pd.read_csv(ESSAYS_CSV)
    if local_df.empty:
        console.print("[yellow]Local CSV is empty — nothing to migrate.[/yellow]")
        return

    console.print(f"[cyan]Found {len(local_df)} essay(s) in local CSV[/cyan]")

    # 3. Load existing Sheet data to avoid duplicates
    sheet_df = backend.load_all_essays()
    console.print(f"[cyan]Sheet currently has {len(sheet_df)} essay(s)[/cyan]")

    # 4. Deduplicate — match on question + essay text
    if not sheet_df.empty and "essay" in sheet_df.columns:
        existing_essays = set(sheet_df["essay"].astype(str).str.strip())
    else:
        existing_essays = set()

    to_migrate = local_df[~local_df["essay"].astype(str).str.strip().isin(existing_essays)]

    if to_migrate.empty:
        console.print("[green]✓ Everything already in Sheets — nothing to migrate.[/green]")
        return

    console.print(f"[cyan]Migrating {len(to_migrate)} new essay(s)...[/cyan]\n")

    # 5. Push each row
    success_count = 0
    for i, row in to_migrate.iterrows():
        row_dict = {h: row.get(h, "") for h in backend.CSV_HEADERS}
        # Fill missing source field if old CSV didn't have it
        if not row_dict.get("source"):
            row_dict["source"] = "migrated"

        success, used = backend.save_essay(row_dict)
        if success and used == "sheets":
            success_count += 1
            console.print(f"  [green]✓[/green] Migrated essay {i+1}/{len(local_df)} "
                          f"({row.get('level','?')} — {row.get('mark','?')}/{row.get('max_marks','?')})")
        else:
            console.print(f"  [red]✗[/red] Failed to migrate essay {i+1} — saved to CSV instead")

    console.print(f"\n[bold green]✓ Migration complete — {success_count} essay(s) added to Sheets[/bold green]")

    new_total = backend.count_essays()
    console.print(f"[cyan]Total essays in Sheets now: {new_total}[/cyan]")


if __name__ == "__main__":
    main()