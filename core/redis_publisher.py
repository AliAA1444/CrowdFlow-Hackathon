from __future__ import annotations

import json
import logging
import time

import redis.asyncio as aioredis

from core.schemas import ZoneUpdate
from core.smoothing import EMACounter

logger = logging.getLogger("crowdflow.redis_publisher")

CHANNEL_ZONE_UPDATES = "zone_updates"
CHANNEL_ZONE_STALE = "zone_stale"


class ZonePublisher:
    """Publishes smoothed zone data to Redis (SET for polling + PUBLISH for streaming).

    Maintains one EMACounter per instance. Callers should create one ZonePublisher
    per zone_id so smoothing state is kept per-zone.
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        zone_id: str,
        alpha: float = 0.3,
    ) -> None:
        self._redis = redis
        self._zone_id = zone_id
        self._ema = EMACounter(alpha=alpha)
        self._last_update: ZoneUpdate | None = None

    async def publish(
        self,
        raw_count: int,
        source: str,
        fps: float,
        zone_area_sqm: float,
        gate_in: int = 0,
        gate_out: int = 0,
    ) -> ZoneUpdate:
        """Smooth raw_count, compute density, and push to Redis.

        Writes to:
          - zone:{zone_id}:latest  (SET, 30 s TTL) for polling consumers
          - zone_updates           (PUBLISH) for real-time subscribers
        """
        smoothed_count = self._ema.update(raw_count)

        area = zone_area_sqm if zone_area_sqm > 0 else 1.0
        density = smoothed_count / area

        update = ZoneUpdate(
            zone_id=self._zone_id,
            density=density,
            raw_count=raw_count,
            smoothed_count=smoothed_count,
            gate_in=gate_in,
            gate_out=gate_out,
            source=source,
            fps=fps,
        )

        payload = json.dumps(update.model_dump())
        key = f"zone:{self._zone_id}:latest"

        await self._redis.set(key, payload, ex=30)
        await self._redis.publish(CHANNEL_ZONE_UPDATES, payload)

        self._last_update = update
        logger.debug("Published zone %s: count=%d smoothed=%.1f", self._zone_id, raw_count, smoothed_count)
        return update

    async def mark_stale(self, reason: str) -> ZoneUpdate | None:
        """Overwrite the zone's Redis key with a stale_fallback payload.

        Preserves the last known smoothed_count — never writes zero — and
        publishes to the zone_stale channel so monitors can react.

        Returns None if no data has been published yet for this zone.
        """
        if self._last_update is None:
            logger.warning("mark_stale called for zone %s but no data published yet", self._zone_id)
            return None

        stale = ZoneUpdate(
            zone_id=self._zone_id,
            density=self._last_update.density,
            raw_count=self._last_update.raw_count,
            smoothed_count=self._last_update.smoothed_count,
            flow_direction=self._last_update.flow_direction,
            flow_magnitude=self._last_update.flow_magnitude,
            gate_in=self._last_update.gate_in,
            gate_out=self._last_update.gate_out,
            source="stale_fallback",
            fps=self._last_update.fps,
            # fresh timestamp so the staleness monitor doesn't re-mark immediately
            timestamp=time.time(),
        )

        payload = json.dumps(stale.model_dump())
        key = f"zone:{self._zone_id}:latest"

        await self._redis.set(key, payload, ex=30)
        await self._redis.publish(CHANNEL_ZONE_STALE, payload)

        logger.warning("Zone %s marked stale: %s", self._zone_id, reason)
        return stale
