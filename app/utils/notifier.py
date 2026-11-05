"""Optional webhook notifier: patrol/evolution failures, model promotion.

Auto-detects the payload format from the configured URL:
  - qyapi.weixin.qq.com  -> WeCom group-bot text message
  - sctapi.ftqq.com      -> ServerChan form post (title/desp)
  - api.telegram.org     -> Telegram sendMessage (needs notify_telegram_chat_id)
  - anything else        -> generic JSON {"title", "content", "level"}

Every send is keyed (event key) and rate-limited by a cooldown so a flapping
patrol cannot spam the channel. Notification failures are logged, never
raised -- it must not break the pipeline it reports about.
"""
from __future__ import annotations

import time

import httpx
from loguru import logger


class Notifier:
    def __init__(self, settings, transport: httpx.BaseTransport | None = None):
        self._url = settings.notify_webhook_url
        self._telegram_chat_id = settings.notify_telegram_chat_id
        self._cooldown_seconds = settings.notify_cooldown_minutes * 60
        self._last_sent: dict[str, float] = {}
        # transport injection lets tests intercept posts without a network
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(transport=self._transport, timeout=10)

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def send(self, title: str, content: str, *, key: str | None = None, level: str = "warn") -> bool:
        if not self._url:
            return False
        key = key or title
        now = time.monotonic()
        last = self._last_sent.get(key, 0.0)
        if now - last < self._cooldown_seconds:
            logger.debug(f"Notification '{key}' suppressed by cooldown.")
            return False

        try:
            response = self._post(title, content, level)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Notification '{key}' failed: {exc}")
            return False
        if response.status_code >= 400:
            logger.warning(f"Notification '{key}' rejected: HTTP {response.status_code}")
            return False
        self._last_sent[key] = now
        logger.info(f"Notification sent: {key}")
        return True

    def _post(self, title: str, content: str, level: str):
        text = f"{title}\n{content}" if content else title
        with self._client() as client:
            if "qyapi.weixin.qq.com" in self._url:
                return client.post(self._url, json={"msgtype": "text", "text": {"content": text}})
            if "sctapi.ftqq.com" in self._url:
                return client.post(self._url, data={"title": title, "desp": content})
            if "api.telegram.org" in self._url:
                payload: dict = {"text": text}
                if self._telegram_chat_id:
                    payload["chat_id"] = self._telegram_chat_id
                return client.post(self._url, json=payload)
            return client.post(self._url, json={"title": title, "content": content, "level": level})
