"""Read-only performance benchmark against the live deployment.

Measures: embedder single vs batch throughput, vector store query latency
(pure-python fallback at scale), end-to-end HTTP endpoints and admin
aggregations. Never writes to the database or the vector store.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8788"


def timed(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def report(label: str, samples: list[float]) -> None:
    print(
        f"{label:<44} avg={statistics.mean(samples):8.1f}ms  "
        f"min={min(samples):8.1f}ms  p95={sorted(samples)[int(len(samples)*0.95)-1]:8.1f}ms"
    )


def http(path: str, token: str | None = None) -> None:
    req = urllib.request.Request(f"{BASE}{path}")
    if token:
        req.add_header("X-Admin-Token", token)
    urllib.request.urlopen(req, timeout=30).read()


def main() -> None:
    from app.config import Settings
    from app.services.container import build_services
    from app.database import create_db_engine, init_db

    settings = Settings()
    engine = create_db_engine(settings.database_url)  # read-only usage below
    init_db(engine)
    services = build_services(settings, engine)
    admin_token = None
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ADMIN_TOKEN="):
            admin_token = line.split("=", 1)[1].strip()
            break

    # ---- embedder: single vs batch (trained encoder, CPU) ----
    queries = [
        "我需要一个终端文件管理器",
        "deep learning library for computer vision",
        "web framework with async support",
        "命令行下载数据的工具",
        "time series database for monitoring",
    ] * 2  # 10 texts
    services.embedding.embed("warmup")  # model warm
    single = timed(lambda: [services.embedding.embed(q) for q in queries], 3)
    report("embed 10 texts (one-by-one)", single)
    batch = timed(lambda: services.embedding.embed_batch(queries), 3)
    report("embed 10 texts (single batch)", batch)

    # ---- chroma query ----
    from app.services.embedding_service import build_project_text
    from app.models.project import Project

    with services.session_factory() as session:
        sample = session.query(Project).filter(Project.analysis_status == "done").first()
    if sample is not None:
        vec = services.embedding.embed(build_project_text(sample))
        report("chroma query top-5 (243 vectors)", timed(lambda: services.vectors.query(vec, top_k=5), 20))

    # ---- memory fallback at scale (pure python, pre-optimization) ----
    import random

    from app.vectorstore.memory_store import MemoryVectorStore

    store = MemoryVectorStore()
    rng = random.Random(42)
    dim = 384
    n = 20000
    ids = [f"p{i}" for i in range(n)]
    vecs = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]
    t0 = time.perf_counter()
    store.upsert(ids, vecs)
    print(f"{'memory store upsert 20k x 384':<44} {((time.perf_counter()-t0)*1000):8.1f}ms")
    q = [rng.uniform(-1, 1) for _ in range(dim)]
    report("memory store query 20k", timed(lambda: store.query(q, top_k=5), 5))

    # ---- e2e HTTP ----
    body = json.dumps({"query": "终端文件管理器", "top_k": 5}).encode()
    def recommend():
        req = urllib.request.Request(f"{BASE}/api/recommend", data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    recommend()
    report("POST /api/recommend (e2e)", timed(recommend, 10))
    report("GET /api/projects?limit=50", timed(lambda: http("/api/projects?limit=50"), 10))
    if admin_token:
        report("GET /api/admin/feedback", timed(lambda: http("/api/admin/feedback", admin_token), 5))
        report("GET /api/admin/digest", timed(lambda: http("/api/admin/digest", admin_token), 5))
        report("GET /api/admin/status", timed(lambda: http("/api/admin/status", admin_token), 5))

    engine.dispose()


if __name__ == "__main__":
    main()
