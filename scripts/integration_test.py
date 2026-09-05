"""Remote integration tests for a deployed GitHub Agent instance.

Run against ANY reachable deployment:
    python scripts/integration_test.py --base https://xxx.trycloudflare.com
    ADMIN_TOKEN=... python scripts/integration_test.py --base http://127.0.0.1:8788

Checks the full external-user journey: health, landing page, docs, semantic
recommendation (zh + en), feedback write -> admin report round-trip, voice
status. Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

RESULTS: list[tuple[str, bool, str]] = []


def call(base: str, path: str, payload: dict | None = None, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, (json.loads(body) if body and body.startswith(("{", "[")) else body)


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name} {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Deployment base URL")
    parser.add_argument("--admin-token", default=os.environ.get("ADMIN_TOKEN", ""))
    parser.add_argument("--seed-keywords", nargs="*", default=["terminal file manager", "deep learning", "web framework"])
    parser.add_argument("--seed-wait-seconds", type=int, default=300)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    admin_headers = {"X-Admin-Token": args.admin_token} if args.admin_token else {}

    print(f"Integration tests against: {base}")

    # 1. health
    status, health = call(base, "/health")
    check("health 200 + ok", status == 200 and health.get("status") == "ok", str(health.get("status")))
    check("trained analyzer active", health["components"]["llm_provider"] == "trained", health["components"]["llm_provider"])
    check("persistent vector store", "Chroma" in health["components"]["vector_store"], health["components"]["vector_store"])

    # 2. landing page + docs
    status, landing = call(base, "/")
    check("landing page", status == 200 and "<html" in landing.lower())
    status, _ = call(base, "/docs")
    check("swagger docs", status == 200)

    # 3. seed data via the autonomous fetch (admin)
    if args.admin_token:
        try:
            call(base, "/api/admin/trigger-fetch", {"keywords": args.seed_keywords}, admin_headers)
        except urllib.error.HTTPError as exc:
            check("trigger fetch", False, f"HTTP {exc.code}")
        deadline = time.time() + args.seed_wait_seconds
        analyzed = 0
        while time.time() < deadline:
            try:
                _, status_body = call(base, "/api/admin/status", headers=admin_headers)
                analyzed = status_body["analyzed"]
                if analyzed > 0 and status_body["vector_count"] > 0 and status_body["pending"] == 0:
                    break
            except Exception:
                pass
            time.sleep(5)
        check("data bootstrapped", analyzed > 0, f"analyzed={analyzed}")

    # 4. recommendations, Chinese and English
    for query, min_results in (("我需要一个终端文件管理器", 1), ("a tool for deep learning", 1)):
        try:
            _, rec = call(base, "/api/recommend", {"query": query, "top_k": 3})
            check(
                f"recommend({query!r})",
                len(rec.get("results", [])) >= min_results
                and all(r["project"]["full_name"] for r in rec["results"]),
                f"{len(rec.get('results', []))} results",
            )
        except urllib.error.HTTPError as exc:
            check(f"recommend({query!r})", False, f"HTTP {exc.code}")

    # 5. feedback round-trip
    try:
        call(base, "/api/feedback", {"query": "integration-test", "vote": "up", "source": "api"})
        check("feedback accepted", True)
        if args.admin_token:
            _, report = call(base, "/api/admin/feedback", headers=admin_headers)
            check(
                "feedback visible in admin report",
                any(f["query"] == "integration-test" for f in report.get("recent", [])),
            )
    except urllib.error.HTTPError as exc:
        check("feedback accepted", False, f"HTTP {exc.code}")

    # 6. voice status (may be disabled on cloud deployments)
    status, voice = call(base, "/api/voice/status")
    check("voice status endpoint", status == 200 and "asr_available" in voice, json.dumps(voice))

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("-" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("ALL INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
