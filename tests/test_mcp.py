"""Tests for the MCP endpoint: handshake, tool listing, tool calls, auth."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.test_api import seed_project_with_embedding


def _rpc(method: str, request_id: int | str | None = 1, params: dict | None = None) -> dict:
    body: dict = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if request_id is not None:
        body["id"] = request_id
    return body


def test_initialize_handshake(client):
    response = client.post("/mcp", json=_rpc("initialize", params={"protocolVersion": "2025-06-18"}))
    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["result"]["protocolVersion"] == "2025-06-18"
    assert body["result"]["serverInfo"]["name"] == "github-agent"
    assert "tools" in body["result"]["capabilities"]


def test_notification_acknowledged_without_body(client):
    response = client.post("/mcp", json=_rpc("notifications/initialized", request_id=None))
    assert response.status_code == 202
    assert response.content == b""


def test_ping_and_unknown_method(client):
    assert client.post("/mcp", json=_rpc("ping")).json()["result"] == {}
    body = client.post("/mcp", json=_rpc("no/such/method")).json()
    assert body["error"]["code"] == -32601


def test_tools_listed(client):
    tools = client.post("/mcp", json=_rpc("tools/list")).json()["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert names == {"recommend_scenario", "research_problem", "list_projects", "get_project"}
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"


def test_research_problem_tool_is_wired(client, monkeypatch):
    def fake_research(problem, include_alternatives=True):
        assert include_alternatives is True
        return {
            "problem": problem,
            "domain": "testing",
            "keywords": ["test kit"],
            "direct": [{"full_name": "acme/testkit", "category": "direct", "reason": "r"}],
            "alternatives": [],
            "summary": "s",
            "cached": False,
        }

    monkeypatch.setattr(client.app.state.services.research, "research", fake_research)
    response = client.post(
        "/mcp",
        json=_rpc("tools/call", params={"name": "research_problem", "arguments": {"problem": "testing tools"}}),
    )
    assert response.status_code == 200
    payload = response.json()["result"]["structuredContent"]
    assert payload["data_source"] == "live-github-index"
    assert payload["direct"][0]["full_name"] == "acme/testkit"


def test_recommend_scenario_tool(client):
    services = client.app.state.services
    seed_project_with_embedding(services, 11, "acme/testkit", "a testing toolkit", ["testing"])

    response = client.post(
        "/mcp",
        json=_rpc("tools/call", params={"name": "recommend_scenario", "arguments": {"query": "testing tools", "top_k": 3}}),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload["data_source"] == "live-github-index"
    assert payload["results"][0]["full_name"] == "acme/testkit"
    # trust fields travel with every recommendation
    first = payload["results"][0]
    for field in ("stars", "license", "maintenance", "pushed_at", "last_indexed_at"):
        assert field in first
    # text content mirrors the structured payload for text-only clients
    assert "acme/testkit" in result["content"][0]["text"]


def test_list_projects_and_get_project_tools(client):
    services = client.app.state.services
    seed_project_with_embedding(services, 12, "acme/lib2", "another toolkit", ["testing"])

    listed = client.post(
        "/mcp",
        json=_rpc("tools/call", params={"name": "list_projects", "arguments": {"language": "Python"}}),
    ).json()["result"]["structuredContent"]
    assert listed["count"] >= 1
    assert listed["projects"][0]["full_name"] == "acme/lib2"

    found = client.post(
        "/mcp",
        json=_rpc("tools/call", params={"name": "get_project", "arguments": {"full_name": "acme/lib2"}}),
    ).json()["result"]["structuredContent"]
    assert found["found"] is True
    assert found["maintenance"] == "unknown"  # no pushed_at recorded in the seed

    missing = client.post(
        "/mcp",
        json=_rpc("tools/call", params={"name": "get_project", "arguments": {"full_name": "nope/none"}}),
    ).json()["result"]["structuredContent"]
    assert missing["found"] is False


def test_unknown_tool_is_a_jsonrpc_error(client):
    body = client.post("/mcp", json=_rpc("tools/call", params={"name": "nope"})).json()
    assert body["error"]["code"] == -32602


@pytest.fixture()
def authed_client(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'mcp.db').as_posix()}",
        scheduler_enabled=False,
        llm_provider="rule",
        llm_fallback_providers=[],
        embedder_provider="tfidf",
        embedding_dimension=64,
        vector_store_backend="memory",
        admin_token="test-admin-token",
        secret_key="test-secret",
        github_fetch_readme=False,
        asr_provider="none",
        tts_provider="none",
        mcp_auth_token="s3cret-mcp-token",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_mcp_auth_required(authed_client):
    assert authed_client.post("/mcp", json=_rpc("ping")).status_code == 401
    assert (
        authed_client.post(
            "/mcp", json=_rpc("ping"), headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = authed_client.post(
        "/mcp", json=_rpc("ping"), headers={"Authorization": "Bearer s3cret-mcp-token"}
    )
    assert ok.status_code == 200
    assert ok.json()["result"] == {}
