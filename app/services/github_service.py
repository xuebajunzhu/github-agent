"""GitHub REST API client with search, README fetching and rate-limit backoff.

Uses httpx directly (instead of PyGithub) for precise control over retries,
rate limits and easy test mocking; the design doc allows either approach.
"""
from __future__ import annotations

import time

import httpx
from loguru import logger

from app.config import Settings
from app.utils.timeutil import parse_github_datetime


class GitHubServiceError(Exception):
    pass


class GitHubAPIError(GitHubServiceError):
    pass


class GitHubRateLimitError(GitHubServiceError):
    pass


class GitHubService:
    def __init__(
        self,
        settings: Settings,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self._settings = settings
        self._token = token or settings.github_token
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"{settings.app_name}/0.1",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.Client(
            base_url=settings.github_api_base,
            headers=headers,
            timeout=30.0,
            transport=transport,
            follow_redirects=True,
        )

    def search_repositories(
        self,
        keyword: str,
        *,
        language: str | None = None,
        min_stars: int | None = None,
        per_page: int | None = None,
        max_repos: int | None = None,
        sort: str = "stars",
    ) -> list[dict]:
        """Search repositories by keyword, sorted by stars; returns normalized dicts."""
        per_page = per_page or self._settings.github_per_page
        max_repos = max_repos or self._settings.github_max_repos_per_keyword
        min_stars = min_stars if min_stars is not None else self._settings.github_min_stars
        language = language or self._settings.github_language

        query = f"{keyword} stars:>={min_stars}"
        if language:
            query += f" language:{language}"

        repos: list[dict] = []
        page = 1
        while len(repos) < max_repos and page <= 10:
            response = self._request(
                "GET",
                "/search/repositories",
                params={
                    "q": query,
                    "sort": sort,
                    "order": "desc",
                    "per_page": min(per_page, max_repos - len(repos)),
                    "page": page,
                },
            )
            items = response.json().get("items", [])
            if not items:
                break
            repos.extend(self._normalize_repo(item) for item in items)
            page += 1
        return repos[:max_repos]

    def get_repository(self, full_name: str) -> dict:
        response = self._request("GET", f"/repos/{full_name}")
        return self._normalize_repo(response.json())

    def get_readme_excerpt(self, full_name: str) -> str | None:
        try:
            response = self._request(
                "GET",
                f"/repos/{full_name}/readme",
                headers={"Accept": "application/vnd.github.raw+json"},
            )
        except GitHubAPIError as exc:
            logger.debug(f"No README for {full_name}: {exc}")
            return None
        text = response.text
        return text[: self._settings.github_readme_max_chars] if text else None

    def _request(self, method: str, path: str, *, params=None, headers=None) -> httpx.Response:
        delay = self._settings.github_backoff_seconds
        last_error: Exception | None = None
        rate_limited = False
        for attempt in range(self._settings.github_max_retries + 1):
            rate_limited = False
            try:
                response = self._client.request(method, path, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last_error = GitHubAPIError(f"GitHub request to {path} failed: {exc}")
                logger.warning(f"GitHub request error ({path}): {exc}")
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code in (403, 429) and self._is_rate_limited(response):
                rate_limited = True
                wait = self._retry_after_seconds(response, delay)
                logger.warning(
                    f"GitHub rate limited ({path}); retry {attempt + 1}/"
                    f"{self._settings.github_max_retries} in {wait:.1f}s"
                )
                time.sleep(wait)
                delay *= 2
                continue
            if response.status_code in (403, 429):
                raise GitHubAPIError(
                    f"GitHub request to {path} failed with {response.status_code}"
                )
            if response.status_code >= 400:
                raise GitHubAPIError(
                    f"GitHub request to {path} failed with {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return response

        if rate_limited:
            raise GitHubRateLimitError(
                f"GitHub rate limit exceeded for {path} after "
                f"{self._settings.github_max_retries} retries"
            )
        raise last_error or GitHubAPIError(f"GitHub request to {path} failed")

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        if response.headers.get("x-ratelimit-remaining") == "0":
            return True
        return "rate limit" in response.text.lower()

    @staticmethod
    def _retry_after_seconds(response: httpx.Response, default: float) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        reset = response.headers.get("x-ratelimit-reset")
        if reset:
            try:
                return min(max(float(reset) - time.time(), 1.0), 120.0)
            except ValueError:
                pass
        return min(default, 120.0)

    @staticmethod
    def _normalize_repo(item: dict) -> dict:
        return {
            "github_id": item["id"],
            "full_name": item["full_name"],
            "url": item.get("html_url", f"https://github.com/{item['full_name']}"),
            "description": item.get("description"),
            "stars": int(item.get("stargazers_count") or 0),
            "forks": int(item.get("forks_count") or 0),
            "language": item.get("language"),
            "topics": list(item.get("topics") or []),
            "pushed_at": parse_github_datetime(item.get("pushed_at")),
        }

    def close(self) -> None:
        self._client.close()
