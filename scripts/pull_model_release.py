"""Pull the latest gated model release from GitHub and promote it locally.

The nightly GitHub Actions workflow (evolution.yml) trains a challenger on
GitHub's compute, gates it against the golden set, and publishes passing
models as Release assets (tag prefix "model-"). This script closes the loop:

    python scripts/pull_model_release.py                # pull + gate + promote
    GITHUB_REPO=xuebajunzhu/github-agent python scripts/pull_model_release.py

Promotion still runs the local canary + golden gates via EvolutionService, so
a bad release can never reach production even if it passed on the runner.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.container import build_services  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPO", "xuebajunzhu/github-agent"))
    args = parser.parse_args()

    settings = Settings()
    root = Path(settings.trained_model_path).resolve().parent

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        release_resp = client.get(f"https://api.github.com/repos/{args.repo}/releases/latest")
        if release_resp.status_code == 404:
            print("No model releases published yet.")
            return
        release_resp.raise_for_status()
        release = release_resp.json()
        asset = next(
            (a for a in release.get("assets", []) if a["name"] == "model.tar.gz"), None
        )
        if asset is None:
            print(f"Release {release['tag_name']} has no model.tar.gz asset.")
            return
        tag = release["tag_name"]
        print(f"Latest gated model release: {tag} ({release['published_at']})")

        extract_dir = Path(tempfile.mkdtemp(prefix="model-pull-"))
        archive_path = extract_dir / "model.tar.gz"
        with client.stream("GET", asset["browser_download_url"], timeout=600) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as handle:
                for chunk in response.iter_bytes(1 << 16):
                    handle.write(chunk)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        staging = extract_dir / "staging-ci"
        print(f"Downloaded and extracted challenger to {staging}")

    engine = None
    services = build_services(settings, __import__("app.database", fromlist=["create_db_engine"]).create_db_engine(settings.database_url))
    evolution = services.evolution

    if not evolution._canary(staging):
        print("Canary FAILED on the downloaded model; discarding.")
        shutil.rmtree(extract_dir, ignore_errors=True)
        return

    metrics = evolution._evaluate_model(staging)
    current = evolution._evaluate_model(settings.trained_model_path)
    print(f"challenger golden F1={metrics['micro_f1']}  champion F1={current['micro_f1']}")
    print(f"challenger gate failures: {metrics['gate_failures'] or 'none'}")

    if metrics["gate_failures"] or metrics["micro_f1"] <= current["micro_f1"]:
        print("Challenger did not beat the deployed model; discarding.")
        shutil.rmtree(extract_dir, ignore_errors=True)
        return

    engine = services.engine
    version = tag.replace("model-", "gh-")
    with evolution._swap_lock:
        live = Path(settings.trained_model_path)
        archive_dir = root / "archive" / version
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        if live.exists():
            shutil.move(str(live), str(archive_dir))
        shutil.move(str(staging), str(live))
        (live / "version.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "source": f"{args.repo} release {tag}",
                    "golden_micro_f1": metrics["micro_f1"],
                    "golden_recall": metrics["recall"],
                    "lineage": str(archive_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for analyzer in getattr(services.analysis, "analyzers", []) or []:
            if getattr(analyzer, "name", "") == "trained":
                analyzer._loaded = None
                analyzer._stamp = None
                analyzer._ensure_loaded()
        embedder = getattr(services.embedding, "embedder", None)
        if embedder is not None and hasattr(embedder, "_reload_if_swapped"):
            embedder._reload_if_swapped()
        reembedded = evolution._reembed_all()
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"PROMOTED {version}; re-embedded {reembedded} vectors.")


if __name__ == "__main__":
    main()
