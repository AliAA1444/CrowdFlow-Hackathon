from __future__ import annotations

"""CrowdFlow — Crowd density simulator.

Replaces real camera feeds for the MVP. Models realistic occupancy
patterns per zone type across a 90-minute match and publishes zone
metrics to Redis for agents and dashboard.

Run standalone:  python simulation/crowd_simulator.py
"""

import asyncio
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import redis.asyncio as aioredis

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    REDIS_URL,
    SAFETY_MAX_DENSITY,
    ZONES,
    classify_congestion,
)

# ── Channels ─────────────────────────────────────────────────────────────────
CH_ZONE_METRICS = "broadcast:zone_metrics"
CH_SIM_CONTROL = "sim:control"

# ── Tunables ─────────────────────────────────────────────────────────────────
SPEED_MULTIPLIER = float(10)          # 10x → 90 min in 9 real min
MATCH_DURATION = 90                   # minutes
PUBLISH_INTERVAL = 2.0                # real seconds between zone broadcasts
NOISE_SIGMA = 0.04                    # Gaussian noise σ (fraction of capacity)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CROWD] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crowd_simulator")


# ── Zone occupancy curves ───────────────────────────────────────────────────

def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _zone_occupancy_fraction(zone_id: str, match_minute: float) -> float:
    """Return a 0-1 occupancy fraction for *zone_id* at *match_minute*.

    Zone behaviour:
      - Stands: full during play (~0.90), dip at halftime (~0.30)
      - Concourses: spike at halftime (~0.85), moderate pre/post-match
      - Gates/entrances: spike pre-match and post-match, quiet during play
    """
    m = match_minute

    if "stand" in zone_id:
        # High during play, dip at halftime
        base = 0.85
        halftime_dip = -0.55 * _gauss(m, 47, 5)
        pre_fill = -0.50 * _gauss(m, 0, 6)   # still filling
        post_empty = -0.60 * _gauss(m, 92, 5) # emptying
        return max(0.05, base + halftime_dip + pre_fill + post_empty)

    if "concourse" in zone_id:
        # Spike at halftime, moderate otherwise
        base = 0.15
        halftime_spike = 0.70 * _gauss(m, 47, 5)
        pre_match = 0.35 * _gauss(m, 3, 6)
        post_match = 0.40 * _gauss(m, 92, 5)
        return min(0.95, base + halftime_spike + pre_match + post_match)

    if "entrance" in zone_id or "gate" in zone_id:
        # Peaks pre-match and post-match
        base = 0.05
        pre = 0.80 * _gauss(m, 0, 8)
        post = 0.85 * _gauss(m, 93, 6)
        halftime_trickle = 0.20 * _gauss(m, 47, 4)
        return min(0.95, base + pre + post + halftime_trickle)

    # Fallback for any unknown zone type
    return 0.30


# ── Simulator state ─────────────────────────────────────────────────────────

class CrowdSimulator:
    def __init__(self) -> None:
        self.match_minute: float = 0.0
        self.running: bool = True
        self.redis: aioredis.Redis | None = None
        # Emergency overrides: zone_id → forced density (persons/m²)
        self.emergency_overrides: dict[str, float] = {}

    # ── Redis helpers ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self.redis.ping()
            log.info("Connected to Redis at %s", REDIS_URL)
        except Exception as exc:
            log.error("Redis connection failed: %s", exc)
            raise

    async def _publish(self, channel: str, data: dict) -> None:
        try:
            await self.redis.publish(channel, json.dumps(data))
        except Exception as exc:
            log.error("Publish to %s failed: %s", channel, exc)

    async def _store(self, key: str, data: dict, ttl: int = 30) -> None:
        try:
            await self.redis.set(key, json.dumps(data), ex=ttl)
        except Exception as exc:
            log.error("Redis SET %s failed: %s", key, exc)

    # ── Control listener ─────────────────────────────────────────────────────

    async def _listen_control(self) -> None:
        sub = self.redis.pubsub()
        try:
            await sub.subscribe(CH_SIM_CONTROL)
        except Exception as exc:
            log.error("Could not subscribe to %s: %s", CH_SIM_CONTROL, exc)
            return

        log.info("Listening on %s", CH_SIM_CONTROL)
        try:
            async for msg in sub.listen():
                if msg["type"] != "message":
                    continue
                try:
                    payload = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._handle_control(payload)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("Control listener error: %s", exc)
        finally:
            try:
                await sub.unsubscribe(CH_SIM_CONTROL)
                await sub.aclose()
            except Exception:
                pass

    async def _handle_control(self, payload: dict) -> None:
        action = payload.get("action")
        if action == "halftime_jump":
            self.match_minute = 45.0
            log.info("⏩ Jumped to halftime (min 45)")
        elif action == "reset":
            self.match_minute = 0.0
            self.emergency_overrides.clear()
            log.info("🔄 Simulator reset")
        elif action == "emergency":
            zone_id = payload.get("zone_id")
            density = float(payload.get("density", SAFETY_MAX_DENSITY + 0.2))
            if zone_id and zone_id in ZONES:
                self.emergency_overrides[zone_id] = density
                log.info(
                    "🚨 EMERGENCY override: %s → %.1f p/m²",
                    zone_id, density,
                )
        elif action == "clear_emergency":
            zone_id = payload.get("zone_id")
            if zone_id and zone_id in self.emergency_overrides:
                del self.emergency_overrides[zone_id]
                log.info("✅ Cleared emergency for %s", zone_id)
            elif zone_id is None:
                self.emergency_overrides.clear()
                log.info("✅ Cleared all emergencies")

    # ── Zone metric computation ──────────────────────────────────────────────

    def _compute_zone(self, zone_id: str, zone_cfg: dict) -> dict:
        capacity = zone_cfg["capacity"]
        area = zone_cfg["w"] * zone_cfg["h"]

        if zone_id in self.emergency_overrides:
            density = self.emergency_overrides[zone_id]
            occupancy = int(density * area)
        else:
            frac = _zone_occupancy_fraction(zone_id, self.match_minute)
            # Add Gaussian noise
            frac += random.gauss(0, NOISE_SIGMA)
            frac = max(0.01, min(frac, 1.0))
            occupancy = int(frac * capacity)
            density = occupancy / area if area > 0 else 0.0

        occupancy_pct = round(occupancy / capacity * 100, 1) if capacity else 0.0
        congestion = classify_congestion(occupancy_pct)

        return {
            "zone_id": zone_id,
            "label": zone_cfg["label"],
            "occupancy": occupancy,
            "capacity": capacity,
            "utilization": round(occupancy / capacity, 3) if capacity else 0,
            "density": round(density, 3),
            "congestion": congestion,
            "area_m2": area,
            "match_minute": round(self.match_minute, 1),
            "timestamp": time.time(),
        }

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.connect()

        control_task = asyncio.create_task(self._listen_control())
        log.info(
            "Crowd Simulator started — speed %.0fx, publish every %.1fs",
            SPEED_MULTIPLIER, PUBLISH_INTERVAL,
        )

        try:
            while self.running and self.match_minute <= MATCH_DURATION:
                zone_metrics = []
                for zone_id, zone_cfg in ZONES.items():
                    metric = self._compute_zone(zone_id, zone_cfg)
                    zone_metrics.append(metric)
                    # Store latest state per zone for API queries
                    await self._store(f"zone:{zone_id}", metric)

                broadcast = {
                    "match_minute": round(self.match_minute, 1),
                    "timestamp": time.time(),
                    "zones": zone_metrics,
                }
                await self._publish(CH_ZONE_METRICS, broadcast)

                # Store full snapshot for API
                await self._store("zones:latest", broadcast)

                # Log summary every ~10 sim-minutes
                sim_minutes_per_tick = (PUBLISH_INTERVAL * SPEED_MULTIPLIER) / 60.0
                log_every = max(1, int(10 / sim_minutes_per_tick))
                tick_count = int(self.match_minute / sim_minutes_per_tick)
                if tick_count % log_every == 0:
                    densities = {m["zone_id"]: m["density"] for m in zone_metrics}
                    max_zone = max(densities, key=densities.get)
                    log.info(
                        "Min %4.1f | zones: %s | peak: %s %.2f p/m²",
                        self.match_minute,
                        " ".join(
                            f"{zid[:6]}={d:.1f}" for zid, d in densities.items()
                        ),
                        max_zone,
                        densities[max_zone],
                    )

                # Advance match clock
                sim_seconds_per_tick = PUBLISH_INTERVAL * SPEED_MULTIPLIER
                self.match_minute += sim_seconds_per_tick / 60.0

                await asyncio.sleep(PUBLISH_INTERVAL)

        except asyncio.CancelledError:
            log.info("Simulator cancelled")
        finally:
            control_task.cancel()
            try:
                await control_task
            except asyncio.CancelledError:
                pass
            log.info("Match ended at minute %.1f", self.match_minute)
            if self.redis:
                await self.redis.aclose()


# ── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    sim = CrowdSimulator()
    await sim.run()


if __name__ == "__main__":
    asyncio.run(main())
