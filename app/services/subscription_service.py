"""Subscription service: watch saved scenarios for NEW matching projects.

Each round (scheduled; matches the fetch cadence so fresh ingests are
visible): for every active subscription, run the scenario through semantic
search, keep hits at/above the subscription's score threshold, and notify
the operator's webhook when projects that were not part of the previous
notification appear. The first round after creation reports the current
matches as a baseline (and is remembered, so the next round only reports
what's genuinely new).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.user import Subscription
from app.utils.timeutil import utcnow


@dataclass
class SubscriptionReport:
    started_at: str
    finished_at: str = ""
    checked: int = 0
    notified: int = 0
    new_matches: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class SubscriptionService:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        recommend,
        notifier=None,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._recommend = recommend
        self._notifier = notifier
        self.last_report: dict | None = None
        self.run_count = 0

    def run_round(self, subscription_id: int | None = None) -> dict:
        report = SubscriptionReport(started_at=utcnow().isoformat())
        self.run_count += 1
        try:
            with self._session_factory() as session:
                query = session.query(Subscription).filter(Subscription.active == 1)
                if subscription_id is not None:
                    query = query.filter(Subscription.id == subscription_id)
                subscriptions = query.all()
            report.checked = len(subscriptions)
            for subscription in subscriptions:
                try:
                    new_matches = self._check_one(subscription)
                    report.new_matches += new_matches
                    if new_matches:
                        report.notified += 1
                except Exception as exc:  # noqa: BLE001  # one bad subscription never stops the round
                    report.errors.append(f"subscription {subscription.id}: {exc}")
                    logger.error(f"Subscription {subscription.id} check failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Subscription round failed")
        report.finished_at = utcnow().isoformat()
        self.last_report = report.as_dict()
        if report.new_matches or report.errors:
            logger.info(f"Subscription round: {self.last_report}")
        else:
            logger.debug(f"Subscription round #{self.run_count}: {report.checked} checked, no new matches.")
        return self.last_report

    def _check_one(self, subscription: Subscription) -> int:
        hits = self._recommend.recommend(subscription.query, top_k=subscription.top_k)
        matches = [
            (hit.project.id, hit.project.full_name, hit.score)
            for hit in hits
            if hit.score >= subscription.min_score
        ]
        current_ids = [project_id for project_id, _, _ in matches]
        previous_ids = set(subscription.last_seen_ids or [])
        fresh = [(pid, name, score) for pid, name, score in matches if pid not in previous_ids]

        with self._session_factory() as session:
            stored = session.get(Subscription, subscription.id)
            stored.last_seen_ids = current_ids
            stored.last_notified_at = utcnow()
            session.commit()

        if fresh and self._notifier is not None:
            lines = [
                f"  {name} (score {score:.2f}) https://github.com/{name}"
                for _, name, score in fresh[:5]
            ]
            self._notifier.send(
                f"[github-agent] 订阅「{subscription.query[:30]}」新增 {len(fresh)} 个匹配项目",
                "\n".join(lines),
                key=f"subscription-{subscription.id}",
                level="info",
            )
        if fresh:
            logger.info(
                f"Subscription {subscription.id} ({subscription.query[:30]!r}): "
                f"{len(fresh)} new of {len(matches)} matches."
            )
        return len(fresh)
