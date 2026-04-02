from __future__ import annotations

import asyncio
import json
import logging
import time

import redis.asyncio as aioredis

from core.schemas import ZoneUpdate

logger = logging.getLogger("crowdflow.staleness")

CHANNEL_ZONE_STALE = "zone_stale"
STALE_KEY_TTL = 30  # seconds


class StalenessMonitor:
    """Async background task that detects and marks stale zone data in Redis.

    Every `check_interval` seconds it scans all ``zone:*:latest`` keys.
    Any key whose payload timestamp is older than `stale_threshold` seconds is
    overwritten with a ``stale_fallback`` ZoneUpdate that preserves the last
    known smoothed_count (never zeroed out), and a ``zone_stale`` event is
    published.
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        check_interval: float = 2.0,
        stale_threshold: float = 3.0,
    ) -> None:
        self._redis = redis
        self._check_interval = check_interval
        self._stale_threshold = stale_threshold

    async def run(self) -> None:
        """Run the staleness-check loop. Cancel the task to stop it."""
        logger.info(
            "StalenessMonitor started (interval=%.1fs, threshold=%.1fs)",
            self._check_interval,
            self._stale_threshold,
        )
        try:
            while True:
                await asyncio.sleep(self._check_interval)
                await self._scan_and_mark()
        except asyncio.CancelledError:
            logger.info("StalenessMonitor stopped")
            raise

    async def _scan_and_mark(self) -> None:
        """Scan all zone keys and mark any stale ones."""
        now = time.time()

        async for key in self._redis.scan_iter(match="zone:*:latest"):
            raw = await self._redis.get(key)
            if raw is None:
                continue

            try:
                payload: dict = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON for key %s", key)
                continue

            ts = payload.get("timestamp")
            if ts is None:
                # Legacy payload with no timestamp — treat as stale.
                ts = 0.0

            age = now - ts
            if age <= self._stale_threshold:
                continue

            # Zone is stale — build a stale_fallback ZoneUpdate.
            zone_id = key.removeprefix("zone:").removesuffix(":latest")

            # Resolve smoothed_count from whatever field the payload uses.
            # New payloads have "smoothed_count"; legacy ones use "occupancy" or "person_count".
            smoothed_count = float(
                payload.get("smoothed_count")
                or payload.get("occupancy")
                or payload.get("person_count")
                or 0.0
            )

            density = float(payload.get("density") or payload.get("density_per_m2") or 0.0)
            fps = float(payload.get("fps") or 0.0)

            stale_update = ZoneUpdate(
                zone_id=zone_id,
                density=density,
                raw_count=int(smoothed_count),
                smoothed_count=smoothed_count,
                source="stale_fallback",
                fps=fps,
                timestamp=now,
            )

            stale_payload = json.dumps(stale_update.model_dump())
            await self._redis.set(key, stale_payload, ex=STALE_KEY_TTL)

            stale_event = json.dumps({
                "zone_id": zone_id,
                "last_seen": ts,
                "stale_since": now,
            })
            await self._redis.publish(CHANNEL_ZONE_STALE, stale_event)

            logger.warning(
                "Zone %s is STALE (last seen %.1fs ago) — overwriting with stale_fallback",
                zone_id,
                age,
            )
