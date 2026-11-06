"""Restore the live database / chroma dir from backup artifacts.

CONTROLLED and GUARDED: refuses while the API service is running, snapshots
the current live state first (reversible), stages + integrity-checks before
replacing, and cleans stale WAL/SHM sidecars.

    .venv/Scripts/python.exe scripts/restore_backup.py            # newest backups
    .venv/Scripts/python.exe scripts/restore_backup.py --db backups/db-20261107-030000.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.services.restore_service import RestoreRefused, RestoreService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="db backup path (default: newest db-*.db)")
    parser.add_argument("--chroma", default=None, help="chroma tarball path (default: newest)")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = parser.parse_args()

    service = RestoreService(Settings())

    db_backup = Path(args.db) if args.db else None
    chroma_backup = Path(args.chroma) if args.chroma else None

    if not args.yes:
        answer = input(
            "This will REPLACE the live database and chroma dir from backups "
            f"(current state is snapshotted first). Type RESTORE to continue: "
        )
        if answer.strip() != "RESTORE":
            print("Aborted.")
            return 1

    try:
        report = service.restore_to_live(db_backup=db_backup, chroma_backup=chroma_backup)
    except RestoreRefused as exc:
        print(f"REFUSED: {exc}")
        return 1

    import json

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["restored"] else 1


if __name__ == "__main__":
    sys.exit(main())
