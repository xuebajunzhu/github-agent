"""Symmetric encryption helpers for storing GitHub access tokens at rest."""
from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


class TokenDecryptionError(Exception):
    pass


def derive_key(secret: str) -> str:
    """Derive a stable Fernet key from the application secret key."""
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


class TokenCipher:
    def __init__(self, fernet_key: str | None, secret_key: str):
        self._fernet = Fernet(fernet_key or derive_key(secret_key))

    def encrypt(self, token: str) -> str:
        return self._fernet.encrypt(token.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError) as exc:
            raise TokenDecryptionError(
                "Stored token cannot be decrypted (wrong FERNET_KEY or SECRET_KEY?)"
            ) from exc
