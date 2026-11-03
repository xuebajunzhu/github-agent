"""Time helpers. All timestamps are stored as naive UTC in the database."""
from __future__ import annotations

from datetime import datetime, timezone

# Maintenance tiers (days since the repo was last pushed to):
#   active <= 180d < quiet <= 730d < stale
_MAINTENANCE_TIERS: tuple[tuple[int, str, float], ...] = (
    (180, "active", 1.0),
    (730, "quiet", 0.93),
    (None, "stale", 0.85),
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def maintenance_status(pushed_at: datetime | None) -> str:
    """Human-readable maintenance tier from the repo's last push time.

    Unknown (None) deliberately reports "unknown" instead of "stale": missing
    data must not be punished, it must be surfaced.
    """
    if pushed_at is None:
        return "unknown"
    days = (utcnow() - pushed_at).days
    for threshold, label, _factor in _MAINTENANCE_TIERS:
        if threshold is None or days <= threshold:
            return label
    return "stale"


def maintenance_decay(pushed_at: datetime | None) -> float:
    """Multiplicative score penalty for long-unmaintained repos (1.0 = none)."""
    if pushed_at is None:
        return 1.0
    days = (utcnow() - pushed_at).days
    for threshold, _label, factor in _MAINTENANCE_TIERS:
        if threshold is None or days <= threshold:
            return factor
    return 0.85
