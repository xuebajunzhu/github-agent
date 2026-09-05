"""GitHub OAuth login routes (authlib, lazy-imported)."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.models.user import User
from app.schemas.user import TokenResponse, UserOut
from app.utils.jwt import encode as create_token

router = APIRouter(prefix="/auth/github", tags=["auth"])

GITHUB_API_BASE = "https://api.github.com"
TOKEN_EXPIRES_SECONDS = 7 * 24 * 3600


def _get_oauth(request: Request):
    settings = request.app.state.settings
    if not settings.is_github_oauth_configured:
        raise HTTPException(
            status_code=503,
            detail="GitHub OAuth is not configured. Set GITHUB_CLIENT_ID and "
            "GITHUB_CLIENT_SECRET (see .env.example).",
        )
    try:
        from authlib.integrations.starlette.client import OAuth
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="authlib is not installed; run: pip install -r requirements.txt",
        ) from exc

    if not hasattr(request.app.state, "oauth"):
        oauth = OAuth()
        oauth.register(
            "github",
            client_id=settings.github_client_id,
            client_secret=settings.github_client_secret,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url=f"{GITHUB_API_BASE}/",
            client_kwargs={"scope": "read:user"},
        )
        request.app.state.oauth = oauth
    return request.app.state.oauth


@router.get("/login")
async def github_login(request: Request):
    oauth = _get_oauth(request)
    settings = request.app.state.settings
    return await oauth.github.authorize_redirect(request, settings.github_redirect_uri)


@router.get("/callback", response_model=TokenResponse)
async def github_callback(request: Request):
    oauth = _get_oauth(request)
    settings = request.app.state.settings
    try:
        token = await oauth.github.authorize_access_token(request)
    except Exception as exc:  # noqa: BLE001  # authlib raises various exception types
        raise HTTPException(status_code=401, detail=f"GitHub OAuth failed: {exc}") from exc
    access_token = token.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="GitHub OAuth did not return an access token")

    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE,
        headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"},
        timeout=30.0,
    ) as client:
        profile_response = await client.get("/user")
        profile_response.raise_for_status()
        profile = profile_response.json()

    services = request.app.state.services
    with services.session_factory() as session:
        user = (
            session.query(User).filter_by(github_user_id=profile["id"]).one_or_none()
        )
        encrypted_token = services.token_cipher.encrypt(access_token)
        if user is None:
            user = User(
                github_user_id=profile["id"],
                username=profile.get("login"),
                access_token_encrypted=encrypted_token,
            )
            session.add(user)
        else:
            user.username = profile.get("login")
            user.access_token_encrypted = encrypted_token
        session.commit()
        user_out = UserOut(id=user.id, username=user.username)

    jwt = create_token(
        {"sub": str(user_out.id), "username": user_out.username or ""},
        settings.secret_key,
        expires_in_seconds=TOKEN_EXPIRES_SECONDS,
    )
    return TokenResponse(access_token=jwt, user=user_out)
