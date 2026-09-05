"""Minimal HS256 JWT implementation (stdlib only) for login sessions."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


class JWTError(Exception):
    pass


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def encode(payload: dict, secret: str, expires_in_seconds: int = 7 * 24 * 3600) -> str:
    body = dict(payload)
    now = int(time.time())
    body.setdefault("iat", now)
    body["exp"] = now + expires_in_seconds
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = "{0}.{1}".format(
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url_encode(json.dumps(body, separators=(",", ":")).encode("utf-8")),
    )
    signature = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode(token: str, secret: str) -> dict:
    try:
        header_b64, body_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise JWTError("Malformed token") from exc
    signing_input = f"{header_b64}.{body_b64}"
    expected = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
        raise JWTError("Invalid signature")
    try:
        payload = json.loads(_b64url_decode(body_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise JWTError("Invalid payload") from exc
    if payload.get("exp") is not None and payload["exp"] < time.time():
        raise JWTError("Token expired")
    return payload
