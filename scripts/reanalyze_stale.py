"""Batch re-analyze stored projects with the current LLM chain + rebuild vectors.

The DeepSeek upgrade only re-analyzed newly fetched projects; the pre-existing
index keeps summaries from the weaker era, which caps BGE-M3 retrieval quality
(ai_summary is part of the indexed text). This script:

  1. selects done projects analyzed before --before (default: all of them),
  2. refreshes each from GitHub (license/stars/pushed_at backfill),
  3. requeues them (pending + embedding_id=None + attempts reset),
  4. drives process_backlog in rounds until the queue drains.

Dry-run by default; pass --execute to actually run. Resumable: interrupted
runs leave the remaining projects in 'pending', rerun continues.

  .venv/Scripts/python.exe scripts/reanalyze_stale.py                      # dry run
  .venv/Scripts/python.exe scripts/reanalyze_stale.py --execute \
      --before "2026-11-03 18:20:00"                                       # deepseek-era cutoff
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from app.config import Settings  # noqa: E402
from app.database import create_db_engine, init_db  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.services.container import build_services  # noqa: E402
from app.services.github_service import GitHubAPIError, GitHubRateLimitError  # noqa: E402
from app.services.ingest_service import _apply_repo_data  # noqa: E402
from app.utils.timeutil import utcnow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually run (default: dry-run)")
    parser.add_argument("--before", default=None, help="only projects with last_analyzed_at < this UTC ISO time")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of requeued projects")
    parser.add_argument("--max-minutes", type=int, default=60, help="pipeline driving deadline")
    parser.add_argument("--round-sleep", type=int, default=10, help="seconds between drain rounds")
    args = parser.parse_args()

    settings = Settings()
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    services = build_services(settings, engine)

    # ---- select stale ----
    with services.session_factory() as session:
        query = session.query(Project).filter(Project.analysis_status == "done")
        if args.before:
            query = query.filter(
                Project.last_analyzed_at.is_(None) | (Project.last_analyzed_at < args.before)
            )
        selected = query.order_by(Project.last_analyzed_at.asc().nulls_first()).all()
        if args.limit:
            selected = selected[: args.limit]
    print(f"Selected {len(selected)} stale projects"
          + (f" (last_analyzed_at < {args.before})" if args.before else "")
          + (f", capped at {args.limit}" if args.limit else ""))
    if not selected:
        print("Nothing to do.")
        return
    for project in selected[:5]:
        print(f"  e.g. {project.full_name} (last analyzed {project.last_analyzed_at})")

    if not args.execute:
        print("DRY RUN - rerun with --execute to refresh from GitHub, requeue and drain the pipeline.")
        print(f"Estimated cost: ~{len(selected)} DeepSeek calls + ~{len(selected)} GitHub requests "
              f"+ ~{len(selected)} BGE-M3 embeddings.")
        return

    # ---- phase B: GitHub refresh + requeue ----
    requeued = 0
    refresh_failed = 0
    started = time.perf_counter()
    for index, project in enumerate(selected, 1):
        try:
            data = services.github.get_repository(project.full_name)
            _apply_repo_data(project, data)
        except GitHubRateLimitError as exc:
            logger.warning(f"GitHub rate limit at {project.full_name}; keeping the rest queued: {exc}")
            break
        except GitHubAPIError as exc:
            logger.warning(f"Refresh failed for {project.full_name}: {exc}")
            refresh_failed += 1
        # always requeue: the point is regenerating the analysis with the new chain
        project.analysis_status = "pending"
        project.embedding_id = None
        project.analysis_attempts = 0
        with services.session_factory() as session:
            merged = session.merge(project)
            session.commit()
        requeued += 1
        if index % 25 == 0:
            print(f"  requeued {index}/{len(selected)}")
    print(f"Phase B done: requeued {requeued}, refresh_failed {refresh_failed} "
          f"({time.perf_counter() - started:.0f}s)")

    # ---- phase C: drive the pipeline until the queue drains ----
    analyzed_total = 0
    embedded_total = 0
    rounds = 0
    deadline = time.perf_counter() + args.max_minutes * 60
    while time.perf_counter() < deadline:
        with services.session_factory() as session:
            pending_count = (
                session.query(Project).filter(Project.analysis_status == "pending").count()
            )
            unembedded_count = (
                session.query(Project)
                .filter(Project.analysis_status == "done", Project.embedding_id.is_(None))
                .count()
            )
        if pending_count == 0 and unembedded_count == 0:
            break
        stats = services.ingest.process_backlog()
        rounds += 1
        if stats.get("status") == "skipped":
            time.sleep(20)
            continue
        analyzed_total += stats.get("analyzed", 0)
        embedded_total += stats.get("embedded", 0)
        print(
            f"  round {rounds}: analyzed +{stats.get('analyzed', 0)} failed +{stats.get('analysis_failed', 0)} "
            f"embedded +{stats.get('embedded', 0)} (totals: {analyzed_total}/{embedded_total}, "
            f"pending {pending_count}, unembedded {unembedded_count})"
        )
        if stats.get("analyzed", 0) == 0 and stats.get("embedded", 0) == 0:
            time.sleep(args.round_sleep)
    elapsed = time.perf_counter() - started
    print(f"DONE in {elapsed / 60:.1f} min: requeued {requeued}, analyzed {analyzed_total}, "
          f"embedded {embedded_total}, rounds {rounds}, refresh_failed {refresh_failed}")


if __name__ == "__main__":
    main()
