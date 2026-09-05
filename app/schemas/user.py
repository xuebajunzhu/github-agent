from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UserOut(BaseModel):
    id: int
    username: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
