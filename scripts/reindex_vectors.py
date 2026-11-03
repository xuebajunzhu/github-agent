"""Rebuild the entire vector index with the current embedder (one-shot).

Use after switching EMBEDDER_PROVIDER (e.g. to bge-m3): the startup space
fingerprint check wipes the store and parks every project (embedding_id=None);
this script then re-embeds everything in batches instead of waiting for the
patrol's 30-per-run increments.

    .venv\\Scripts\\python.exe scripts\\reindex_vectors.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from app.config import Settings  # noqa: E402
from app.database import create_db_engine, init_db  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.services.container import build_services  # noqa: E402
from app.services.embedding_service import (  # noqa: E402
    build_project_text,
    project_embedding_id,
)
from app.utils.filelock import job_lock  # noqa: E402
from app.vectorstore.base import upsert_batch_safe  # noqa: E402

BATCH = 32


def main() -> None:
    settings = Settings()
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    services = build_services(settings, engine)  # startup also runs the space check
    print(f"embedder={services.embedding.provider_name} dimension={services.embedding.embedder.dimension}")

    reindexed = 0
    started = time.perf_counter()
    with job_lock() as acquired:
        if not acquired:
            print("Another pipeline job is running; try again later.")
            return
        while True:
            with services.session_factory() as session:
                projects = (
                    session.query(Project)
                    .filter(Project.analysis_status == "done", Project.embedding_id.is_(None))
                    .order_by(Project.id)
                    .limit(BATCH)
                    .all()
                )
                if not projects:
                    break
                texts = [build_project_text(project) for project in projects]
                vectors = services.embedding.embed_batch_safe(texts)
                usable = [
                    (project, text, vector)
                    for project, text, vector in zip(projects, texts, vectors)
                    if vector is not None
                ]
                if not usable:
                    logger.error("A whole batch failed to embed; aborting to avoid a loop.")
                    break
                ids = [project_embedding_id(project.id) for project, _, _ in usable]
                ok = upsert_batch_safe(
                    services.vectors,
                    ids,
                    [vector for _, _, vector in usable],
                    [text for _, text, _ in usable],
                    [
                        {"github_id": project.github_id, "full_name": project.full_name}
                        for project, _, _ in usable
                    ],
                )
                for (project, _, _), embedding_id, succeeded in zip(usable, ids, ok):
                    if succeeded:
                        project.embedding_id = embedding_id
                        reindexed += 1
                session.commit()
                print(f"  batch done: +{sum(ok)} ({reindexed} total)")
    elapsed = time.perf_counter() - started
    print(f"Reindexed {reindexed} vectors in {elapsed:.1f}s ({settings.database_url}).")


if __name__ == "__main__":
    main()
