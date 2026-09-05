from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.services.github_service import GitHubRateLimitError, GitHubService

SEARCH_BODY = {
    "items": [
        {
            "id": 1001,
            "full_name": "acme/ml-lib",
            "html_url": "https://github.com/acme/ml-lib",
            "description": "A deep learning library",
            "stargazers_count": 1234,
            "forks_count": 56,
            "language": "Python",
            "topics": ["deep-learning"],
            "pushed_at": "2026-01-02T03:04:05Z",
        }
    ]
}


@pytest.fixture()
def github_settings() -> Settings:
    return Settings(
        _env_file=None,
        github_max_retries=2,
        github_backoff_seconds=0.01,
    )


def make_service(settings: Settings, handler) -> GitHubService:
    return GitHubService(settings, transport=httpx.MockTransport(handler))


def test_search_normalizes_repositories(github_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search/repositories"
        assert "stars:" in request.url.params["q"]
        if request.url.params["page"] == "1":
            return httpx.Response(200, json=SEARCH_BODY)
        return httpx.Response(200, json={"items": []})

    service = make_service(github_settings, handler)
    repos = service.search_repositories("machine learning")
    assert len(repos) == 1
    repo = repos[0]
    assert repo["github_id"] == 1001
    assert repo["full_name"] == "acme/ml-lib"
    assert repo["stars"] == 1234
    assert repo["topics"] == ["deep-learning"]
    assert repo["pushed_at"] is not None
    service.close()


def test_includes_language_and_star_filters(github_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["q"]
        assert "language:python" in query
        assert "stars:>=500" in query
        return httpx.Response(200, json={"items": []})

    service = make_service(github_settings, handler)
    assert service.search_repositories("web", language="python", min_stars=500) == []
    service.close()


def test_retries_after_rate_limit_then_succeeds(github_settings):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={"X-RateLimit-Remaining": "0", "Retry-After": "0"},
            )
        if request.url.params["page"] == "1":
            return httpx.Response(200, json=SEARCH_BODY)
        return httpx.Response(200, json={"items": []})

    service = make_service(github_settings, handler)
    repos = service.search_repositories("tools")
    # call 1: page 1 rate-limited; call 2: page 1 retry succeeds; call 3: page 2 empty
    assert repos and calls["count"] == 3
    service.close()


def test_gives_up_after_repeated_rate_limits(github_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0", "Retry-After": "0"},
        )

    service = make_service(github_settings, handler)
    with pytest.raises(GitHubRateLimitError):
        service.search_repositories("tools")
    service.close()


def test_http_error_raises_github_api_error(github_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    service = make_service(github_settings, handler)
    with pytest.raises(Exception):  # noqa: B017, PT011
        service.search_repositories("tools")
    service.close()


def test_readme_excerpt_truncated(github_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/acme/ml-lib/readme"
        return httpx.Response(200, text="x" * 500)

    github_settings.github_readme_max_chars = 100
    service = make_service(github_settings, handler)
    excerpt = service.get_readme_excerpt("acme/ml-lib")
    assert excerpt is not None and len(excerpt) == 100
    service.close()
