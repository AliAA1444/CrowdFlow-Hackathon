from __future__ import annotations

"""CrowdFlow — Fallback simulator for zones without video files.

On startup, checks which cameras in CAMERA_FEEDS have actual video files
present on disk.  For any zone whose camera file is missing, this module
generates simulated density data using the same Gaussian occupancy curves
as the original crowd_simulator.py — ensuring the demo always works even
when no video files have been placed in data/videos/.

Zones that DO have video files are left alone; camera_streamer.py handles them.

Run standalone:  python inference/fallback_simulator.py
"""

import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import redis.asyncio as aioredis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CAMERA_FEEDS,
    REDIS_URL,
    ZONES,
    classify_congestion,
    get_state_config,
)

# Re-use the Gaussian occupancy curves from the original simulator
from simulation.crowd_simulator import _gauss, _zone_occupancy_fraction

# Matches the channel used by crowd_simulator.py and all agent files
CH_ZONE_METRICS = "broadcast:zone_metrics"

# ── Tunables ─────────────────────────────────────────────────────────────────
SPEED_MULTIPLIER = 10.0    # 10× → 90 min match in 9 real minutes
MATCH_DURATION   = 90.0    # minutes
PUBLISH_INTERVAL = 2.0     # real seconds between publishes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FALLBACK] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fallback_simulator")


class FallbackSimulator:
    """Simulates zone metrics only for zones whose camera video files are absent."""

    def __init__(self) -> None:
        self.match_minute: float  = 0.0
        self.redis: aioredis.Redis | None = None
        self._fallback_zone_ids: list[str] = []
        # Per-zone persisted counts — strictly monotonic during normal play
        self._zone_counts: dict[str, int] = {}

    # ── Startup: discover which zones need simulation ──────────────────────

    def _find_fallback_zones(self) -> list[str]:
        """
        Returns zone_ids that have no reachable video file.

        A zone is 'covered' if at least one camera in CAMERA_FEEDS points
        to it AND that camera's video file exists on disk.
        """
        covered: set[str] = set()
        for cam_id, cfg in CAMERA_FEEDS.items():
            if os.path.exists(cfg["video_file"]):
                covered.add(cfg["zone_id"])
                log.info("Camera %s: video present — zone '%s' covered by camera_streamer", cam_id, cfg["zone_id"])
            else:
                log.info("Camera %s: video missing — zone '%s' will use fallback", cam_id, cfg["zone_id"])

        fallback = [zid for zid in ZONES if zid not in covered]
        return fallback

    # ── Redis helpers ──────────────────────────────────────────────────────

    async def _connect(self) -> None:
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await self.redis.ping()
        log.info("FallbackSimulator connected to Redis at %s", REDIS_URL)

    async def _publish(self, channel: str, data: dict) -> None:
        try:
            await self.redis.publish(channel, json.dumps(data))
        except Exception as exc:
            log.warning("Publish to %s failed: %s", channel, exc)

    async def _store(self, key: str, data: dict, ttl: int = 30) -> None:
        try:
            await self.redis.set(key, json.dumps(data), ex=ttl)
        except Exception as exc:
            log.warning("Redis SET %s failed: %s", key, exc)

    # Zones that flood with exiting fans at end-of-match
    _ENDMATCH_SPIKE_ZONES = {
        "zone_south_gate", "zone_activation_1", "zone_activation_2", "zone_north_parking",
    }

    async def _get_demo_mode(self) -> str:
        """Read the current demo mode from Redis (default: 'normal')."""
        try:
            val = await self.redis.get("demo:mode")
            return val or "normal"
        except Exception:
            return "normal"

    # ── Zone metric computation ────────────────────────────────────────────

    def _compute_zone(self, zone_id: str, zone_cfg: dict, *, endmatch: bool = False) -> dict:
        capacity = int(zone_cfg["capacity"])
        area_m2  = float(zone_cfg.get("area_m2") or
                         (zone_cfg.get("w", 1) * zone_cfg.get("h", 1)))
        prev_count = self._zone_counts.get(zone_id, 0)

        if endmatch and zone_id in self._ENDMATCH_SPIKE_ZONES:
            # Exit zones: fast ramp toward 80-100% capacity
            target = int(random.uniform(0.80, 1.00) * capacity)
            new_count = min(capacity, prev_count + max(50, (target - prev_count) // 4))
        elif endmatch:
            # Interior zones during endmatch: slow drain, floor at 10%
            floor = int(0.10 * capacity)
            new_count = max(floor, prev_count - random.randint(20, 80))
        elif self.match_minute <= 45:
            # Phase 1 (0-45 min): STRICTLY MONOTONIC INCREASE
            # Seed from curve on first tick
            if prev_count == 0:
                target_frac = _zone_occupancy_fraction(zone_id, self.match_minute)
                prev_count  = max(1, int(target_frac * capacity * 0.5))
            ceiling   = int(0.90 * capacity)
            growth    = random.randint(10, 50)
            new_count = min(ceiling, prev_count + growth)
        else:
            # Phase 2 (45-90 min): PERFECTLY FLAT — no changes
            new_count = prev_count

        # Never drop to zero
        new_count = max(new_count, 1)
        self._zone_counts[zone_id] = new_count

        person_count  = new_count
        density       = round(person_count / area_m2, 2) if area_m2 > 0 else 0.0
        occupancy_pct = round(person_count / capacity * 100, 1) if capacity > 0 else 0.0
        congestion_key = classify_congestion(occupancy_pct)
        state          = get_state_config(congestion_key)

        return {
            # ── Legacy fields (agents use these exact names) ─────────────────
            "zone_id":     zone_id,
            "label":       zone_cfg["label"],
            "occupancy":   person_count,
            "capacity":    capacity,
            "utilization": round(person_count / capacity, 3) if capacity else 0.0,
            "density":     density,
            "congestion":  congestion_key,
            "area_m2":     area_m2,
            "match_minute": round(self.match_minute, 1),
            "timestamp":   time.time(),
            # ── Extended fields ───────────────────────────────────────────────
            "person_count":     person_count,
            "density_per_m2":   density,
            "occupancy_pct":    occupancy_pct,
            "congestion_level": congestion_key,
            "congestion_color": state["color"],
            "congestion_icon":  state["icon"],
            "camera_id":        None,     # no real camera
            "inference_ms":     None,
            "detection_boxes":  [],
            "source":           "fallback_simulator",
        }

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._connect()
        self._fallback_zone_ids = self._find_fallback_zones()

        if not self._fallback_zone_ids:
            log.info("All zones have video files — FallbackSimulator running in standby (park loop).")
            # Stay alive; camera_streamer handles everything
            try:
                while True:
                    await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                pass
            finally:
                if self.redis:
                    await self.redis.aclose()
            return

        log.info(
            "FallbackSimulator covering %d zone(s): %s",
            len(self._fallback_zone_ids),
            ", ".join(self._fallback_zone_ids),
        )

        try:
            while True:
                # Clamp match clock — don't reset to avoid discontinuity
                if self.match_minute > MATCH_DURATION:
                    self.match_minute = MATCH_DURATION

                demo_mode = await self._get_demo_mode()
                endmatch  = demo_mode == "endmatch"

                metrics: list[dict] = []
                for zone_id in self._fallback_zone_ids:
                    zone_cfg = ZONES[zone_id]
                    metric   = self._compute_zone(zone_id, zone_cfg, endmatch=endmatch)
                    metrics.append(metric)
                    await self._store(f"zone:{zone_id}:latest", metric, ttl=30)

                broadcast = {
                    "match_minute": round(self.match_minute, 1),
                    "timestamp":    time.time(),
                    "zones":        metrics,
                    "source":       "fallback_simulator",
                }
                await self._publish(CH_ZONE_METRICS, broadcast)

                sim_seconds        = PUBLISH_INTERVAL * SPEED_MULTIPLIER
                self.match_minute += sim_seconds / 60.0
                await asyncio.sleep(PUBLISH_INTERVAL)

        except asyncio.CancelledError:
            log.info("FallbackSimulator cancelled")
        finally:
            log.info("FallbackSimulator stopped at match minute %.1f", self.match_minute)
            if self.redis:
                await self.redis.aclose()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    sim = FallbackSimulator()
    await sim.run()


if __name__ == "__main__":
    asyncio.run(main())
