"""Router registry."""
from fastapi import APIRouter

from app.api.routes import admin, auth, feedback, health, mcp, projects, recommend, research, subscriptions, voice

all_routers: list[APIRouter] = [
    health.router,
    auth.router,
    recommend.router,
    research.router,
    projects.router,
    admin.router,
    voice.router,
    feedback.router,
    mcp.router,
    subscriptions.router,
]
