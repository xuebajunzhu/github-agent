"""Collect GitHub repository metadata as weakly-labeled training data.

Labels come for free: GitHub topics -> tag labels, topics/description ->
use-case labels via app.ml.taxonomy. Paced to stay inside the anonymous
Search API budget (~10 requests/minute).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.services.github_service import GitHubRateLimitError, GitHubService

# Broad keyword coverage across the taxonomy, one search request each.
DEFAULT_KEYWORDS = [
    "machine learning", "deep learning", "reinforcement learning", "nlp", "llm",
    "chatbot", "computer vision", "object detection", "ocr", "speech recognition",
    "web framework", "web application", "react component", "vue", "frontend",
    "backend", "rest api", "graphql", "microservices", "websocket",
    "cli tool", "terminal", "shell", "database", "sql",
    "orm", "nosql", "redis", "search engine", "full-text search",
    "data visualization", "dashboard", "data analysis", "pandas", "jupyter",
    "web scraping", "crawler", "automation", "workflow", "ci cd",
    "devops", "docker", "kubernetes", "monitoring", "terraform",
    "security", "penetration testing", "cryptography", "authentication", "oauth",
    "blockchain", "ethereum", "smart contracts", "game engine", "game",
    "opengl", "audio", "video editing", "ffmpeg", "text to speech",
    "android", "ios", "flutter", "react native", "mobile app",
    "iot", "embedded", "raspberry pi", "mqtt", "firmware",
    "proxy", "vpn", "p2p", "torrent", "download manager",
    "tutorial", "awesome list", "interview questions", "algorithm", "data structures",
    "developer tools", "productivity", "vscode extension", "vim", "git",
    "recommendation system", "robotics", "ros", "simulation", "physics engine",
    "bioinformatics", "finance", "trading", "pdf", "markdown",
    "excel", "file manager", "note taking", "password manager", "ad blocker",
    "streaming", "music player", "photo editor", "e-commerce", "blog",
    "cms", "forum", "email", "calendar", "todo",
    "weather", "map", "gis", "3d printing", "cad",
    "compiler", "programming language", "text editor", "rust", "golang",
    "python library", "javascript library", "typescript", "java library", "swift",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="training/data/raw_repos.jsonl")
    parser.add_argument("--keywords", nargs="*", default=None, help="Override keyword list")
    parser.add_argument(
        "--keywords-file", default=None, help="One keyword per line; overrides --keywords"
    )
    parser.add_argument("--min-stars", type=int, default=30)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=6.5, help="Seconds between search requests")
    parser.add_argument("--language", default=None, help="Optional language filter")
    args = parser.parse_args()
    if args.keywords_file:
        lines = Path(args.keywords_file).read_text(encoding="utf-8").splitlines()
        args.keywords = [line.strip() for line in lines if line.strip()]
    return args


def main() -> None:
    args = parse_args()
    keywords = args.keywords or DEFAULT_KEYWORDS
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings = Settings(_env_file=None)  # anonymous: no token, paced to the limit
    service = GitHubService(settings)

    kept = 0
    skipped = 0
    seen_ids: set[int] = set()
    started = time.time()
    with output_path.open("w", encoding="utf-8") as output:
        for index, keyword in enumerate(keywords, start=1):
            try:
                repos = service.search_repositories(
                    keyword,
                    language=args.language,
                    min_stars=args.min_stars,
                    per_page=args.per_page,
                    max_repos=args.per_page,  # one request per keyword
                )
            except GitHubRateLimitError as exc:
                print(f"[{index}/{len(keywords)}] rate limited at '{keyword}': {exc}")
                print("Waiting 60s before continuing ...")
                time.sleep(60)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[{index}/{len(keywords)}] search failed for '{keyword}': {exc}")
                time.sleep(args.sleep)
                continue

            new_repos = 0
            for repo in repos:
                if repo["github_id"] in seen_ids:
                    continue
                seen_ids.add(repo["github_id"])
                if not (repo["description"] or "").strip() or not repo["topics"]:
                    skipped += 1
                    continue
                record = dict(repo)
                record["pushed_at"] = repo["pushed_at"].isoformat() if repo["pushed_at"] else None
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                new_repos += 1
            if index % 10 == 0 or index == len(keywords):
                elapsed = time.time() - started
                print(
                    f"[{index}/{len(keywords)}] '{keyword}': +{new_repos} | "
                    f"total {kept} (skipped {skipped}) | {elapsed:.0f}s",
                    flush=True,
                )
            time.sleep(args.sleep)

    service.close()
    print(f"Done. {kept} labeled repos written to {output_path} ({skipped} skipped).")


if __name__ == "__main__":
    main()
