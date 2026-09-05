"""Run one real self-evolution cycle with bounded settings (validation run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.database import create_db_engine, init_db
from app.services.container import build_services


def main() -> None:
    settings = Settings(
        evolution_train_epochs=1,
        evolution_max_train_samples=2500,
        evolution_min_new_samples=0,
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    services = build_services(settings, engine)
    report = services.evolution.run_cycle(force=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
