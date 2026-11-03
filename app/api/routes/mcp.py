"""MCP (Model Context Protocol) endpoint: expose the agent as tool infrastructure.

AI coding agents (Claude Code, Cursor, ...) increasingly pick open-source
dependencies autonomously, and they need a live, verified index instead of a
model's training-time memory. This endpoint implements a minimal stateless
subset of the Streamable HTTP transport (JSON-RPC 2.0 over POST, JSON
responses) with the `tools` capability, so the recommendation engine can be
mounted into any MCP client. Listed in the MCP registries, it becomes a
distribution channel for the whole index.

Tools:
  - recommend_scenario(query, top_k)  semantic scenario -> verified projects
  - list_projects(language, min_stars, limit)
  - get_project(full_name)

Auth: set MCP_AUTH_TOKEN to require `Authorization: Bearer <token>`; when
unset the endpoint is open (still covered by the global rate limiter).
"""
from __future__ import annotations

import hmac
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app.utils.timeutil import maintenance_status

router = APIRouter(tags=["mcp"])

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "github-agent"
JSONRPC_PARSE_ERROR = -32700
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_REQUEST = -32602


class MCPError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# --------------------------------------------------------------------- tools


def _project_brief(project) -> dict[str, Any]:
    """Trust-first projection: live GitHub data, never model-invented."""
    return {
        "full_name": project.full_name,
        "url": project.url,
        "description": project.description,
        "stars": project.stars,
        "forks": project.forks,
        "language": project.language,
        "license": project.license,
        "topics": list(project.topics or []),
        "maintenance": maintenance_status(project.pushed_at),
        "pushed_at": project.pushed_at.isoformat() if project.pushed_at else None,
        "summary": project.ai_summary or project.description,
        "use_cases": list(project.ai_use_cases or []),
        "tags": list(project.ai_tags or []),
        "last_indexed_at": project.updated_at.isoformat() if project.updated_at else None,
    }


def _tool_recommend_scenario(services, arguments: dict) -> dict:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise MCPError(JSONRPC_INVALID_REQUEST, "query is required")
    query = query[:300]
    top_k = int(arguments.get("top_k") or 5)
    top_k = max(1, min(top_k, 20))
    hits = services.recommend.recommend(query, top_k=top_k)
    return {
        "query": query,
        "data_source": "live-github-index",
        "results": [
            {
                **_project_brief(hit.project),
                "score": hit.score,
                "dependencies": hit.dependencies,
                "reason": hit.reason,
            }
            for hit in hits
        ],
    }


def _tool_list_projects(services, arguments: dict) -> dict:
    language = arguments.get("language") or None
    min_stars = int(arguments.get("min_stars") or 0)
    limit = max(1, min(int(arguments.get("limit") or 10), 50))
    with services.session_factory() as session:
        from app.models.project import Project

        query = session.query(Project).filter(Project.analysis_status == "done")
        if language:
            query = query.filter(Project.language == language)
        if min_stars > 0:
            query = query.filter(Project.stars >= min_stars)
        rows = query.order_by(Project.stars.desc()).limit(limit).all()
    return {
        "count": len(rows),
        "projects": [_project_brief(project) for project in rows],
    }


def _tool_get_project(services, arguments: dict) -> dict:
    full_name = str(arguments.get("full_name") or "").strip()
    if not full_name:
        raise MCPError(JSONRPC_INVALID_REQUEST, "full_name is required")
    with services.session_factory() as session:
        from app.models.project import Project

        project = (
            session.query(Project).filter(Project.full_name == full_name).one_or_none()
        )
    if project is None:
        return {"found": False, "full_name": full_name}
    return {"found": True, **_project_brief(project)}


def _tool_schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "recommend_scenario",
        "description": (
            "Find open-source projects matching a natural-language scenario "
            "(Chinese and English). Returns live GitHub data (stars, license, "
            "last push, maintenance tier) from a continuously refreshed index -- "
            "never model memory, so no hallucinated project names."
        ),
        "inputSchema": _tool_schema(
            {
                "query": {"type": "string", "description": "Scenario description, e.g. 我需要一个终端文件管理器"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            ["query"],
        ),
    },
    {
        "name": "list_projects",
        "description": "List indexed open-source projects ordered by stars, with live maintenance signals.",
        "inputSchema": _tool_schema(
            {
                "language": {"type": "string", "description": "Filter by programming language"},
                "min_stars": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            [],
        ),
    },
    {
        "name": "get_project",
        "description": "Look up one indexed repository by GitHub full_name (owner/repo).",
        "inputSchema": _tool_schema(
            {"full_name": {"type": "string", "description": "owner/repo"}},
            ["full_name"],
        ),
    },
]

TOOL_HANDLERS = {
    "recommend_scenario": _tool_recommend_scenario,
    "list_projects": _tool_list_projects,
    "get_project": _tool_get_project,
}


# ------------------------------------------------------------------ transport


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


def _rpc_result(request_id: int | str, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(request_id: int | str | None, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_text_result(payload: dict) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
        "structuredContent": payload,
        "isError": False,
    }


def _dispatch(request: JsonRpcRequest, services) -> dict | None:
    """Handle one JSON-RPC call. Returns a response body, or None for notifications."""
    method = request.method
    if method == "notifications/initialized" or method.startswith("notifications/"):
        return None  # notification: acknowledge with 202, no body
    request_id = request.id
    if request_id is None:
        raise MCPError(JSONRPC_INVALID_REQUEST, "method call requires an id")

    if method == "initialize":
        return _rpc_result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
            },
        )
    if method == "ping":
        return _rpc_result(request_id, {})
    if method == "tools/list":
        return _rpc_result(request_id, {"tools": TOOLS})
    if method == "tools/call":
        name = str(request.params.get("name") or "")
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            raise MCPError(JSONRPC_INVALID_REQUEST, f"unknown tool: {name}")
        arguments = request.params.get("arguments") or {}
        return _rpc_result(request_id, _tool_text_result(handler(services, arguments)))
    raise MCPError(JSONRPC_METHOD_NOT_FOUND, f"unknown method: {method}")


def _authorized(request: Request, token: str | None) -> bool:
    if not token:
        return True
    header = request.headers.get("authorization") or ""
    if not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[len("Bearer "):].strip(), token)


@router.post("/mcp")
def mcp_endpoint(payload: JsonRpcRequest, request: Request) -> Response:
    settings = request.app.state.settings
    if not _authorized(request, settings.mcp_auth_token):
        return JSONResponse(
            {"error": "unauthorized: set Authorization: Bearer <MCP_AUTH_TOKEN>"},
            status_code=401,
        )
    services = request.app.state.services
    try:
        body = _dispatch(payload, services)
    except MCPError as exc:
        body = _rpc_error(payload.id, exc.code, exc.message)
    except Exception:  # noqa: BLE001  # tool failures must be MCP-visible, not 500s
        from loguru import logger

        logger.exception("MCP tool call failed")
        body = _rpc_error(payload.id, -32603, "internal error while executing the tool")
    if body is None:  # notification
        return Response(status_code=202)
    return JSONResponse(body)


@router.get("/mcp", include_in_schema=False)
def mcp_endpoint_get() -> Response:
    # Streamable HTTP GET would be the server-initiated SSE channel; this
    # stateless server has nothing to push.
    return JSONResponse(
        {"error": "POST JSON-RPC 2.0 requests to /mcp"}, status_code=405
    )
