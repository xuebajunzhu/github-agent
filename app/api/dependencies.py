"""Shared FastAPI dependencies."""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request

from app.services.container import ServiceContainer


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services


def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    expected = request.app.state.settings.admin_token
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token header.")
