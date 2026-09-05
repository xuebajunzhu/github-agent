from __future__ import annotations

import pytest

from app.utils.jwt import JWTError, decode, encode


def test_roundtrip():
    token = encode({"sub": "42", "username": "alice"}, "secret")
    payload = decode(token, "secret")
    assert payload["sub"] == "42"
    assert payload["username"] == "alice"


def test_tampered_token_rejected():
    token = encode({"sub": "42"}, "secret")
    tampered = token[:-4] + "AAAA"
    with pytest.raises(JWTError):
        decode(tampered, "secret")


def test_wrong_secret_rejected():
    token = encode({"sub": "42"}, "secret-a")
    with pytest.raises(JWTError):
        decode(token, "secret-b")


def test_expired_token_rejected():
    token = encode({"sub": "42"}, "secret", expires_in_seconds=-10)
    with pytest.raises(JWTError):
        decode(token, "secret")
