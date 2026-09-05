"""Router registry."""
from fastapi import APIRouter

from app.api.routes import admin, auth, feedback, health, projects, recommend, voice

all_routers: list[APIRouter] = [
    health.router,
    auth.router,
    recommend.router,
    projects.router,
    admin.router,
    voice.router,
    feedback.router,
]
