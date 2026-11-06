"""Subscription management (operator-level: notifications go to the operator's
webhook, so creation is admin-gated like the rest of the ops surface)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker

from app.api.dependencies import get_services, require_admin
from app.models.user import Subscription
from app.services.container import ServiceContainer

router = APIRouter(
    prefix="/api/subscriptions", tags=["subscriptions"], dependencies=[Depends(require_admin)]
)


class SubscriptionIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


class SubscriptionOut(BaseModel):
    id: int
    query: str
    min_score: float
    top_k: int
    active: bool
    last_seen_count: int
    last_notified_at: str | None

    @classmethod
    def of(cls, subscription: Subscription) -> "SubscriptionOut":
        return cls(
            id=subscription.id,
            query=subscription.query,
            min_score=subscription.min_score,
            top_k=subscription.top_k,
            active=bool(subscription.active),
            last_seen_count=len(subscription.last_seen_ids or []),
            last_notified_at=subscription.last_notified_at.isoformat()
            if subscription.last_notified_at
            else None,
        )


@router.get("")
def list_subscriptions(services: ServiceContainer = Depends(get_services)) -> list[SubscriptionOut]:
    with services.session_factory() as session:
        rows = (
            session.query(Subscription)
            .filter(Subscription.active == 1)
            .order_by(Subscription.id.desc())
            .all()
        )
    return [SubscriptionOut.of(row) for row in rows]


@router.post("", response_model=SubscriptionOut)
def create_subscription(
    payload: SubscriptionIn, services: ServiceContainer = Depends(get_services)
) -> SubscriptionOut:
    with services.session_factory() as session:
        exists = (
            session.query(Subscription)
            .filter(Subscription.query == payload.query, Subscription.active == 1)
            .one_or_none()
        )
        if exists is not None:
            raise HTTPException(status_code=409, detail="该订阅已存在")
        subscription = Subscription(
            query=payload.query,
            min_score=payload.min_score,
            top_k=payload.top_k,
            active=1,
            last_seen_ids=None,  # first round records the baseline
        )
        session.add(subscription)
        session.commit()
        return SubscriptionOut.of(subscription)


@router.delete("/{subscription_id}")
def delete_subscription(
    subscription_id: int, services: ServiceContainer = Depends(get_services)
) -> dict:
    with services.session_factory() as session:
        deleted = (
            session.query(Subscription)
            .filter(Subscription.id == subscription_id)
            .update({"active": 0})
        )
        session.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"ok": True}


@router.post("/check")
def check_subscriptions(
    services: ServiceContainer = Depends(get_services), subscription_id: int | None = None
) -> dict:
    """Run one watch round now (all active subscriptions, or one by id)."""
    report = services.subscription.run_round(subscription_id=subscription_id)
    return report
