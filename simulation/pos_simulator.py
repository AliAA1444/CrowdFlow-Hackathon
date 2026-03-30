from __future__ import annotations

"""CrowdFlow — Point-of-Sale transaction simulator.

Models a 90-minute football match demand curve and generates realistic
concession transactions per stall. Publishes to Redis for downstream
agents and the dashboard.

Run standalone:  python simulation/pos_simulator.py
"""

import asyncio
import json
import logging
import math
import random
import sys
import time
import uuid
from pathlib import Path

import redis.asyncio as aioredis

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CONCESSION_STALLS,
    MENU_ITEMS,
    REDIS_URL,
)

# ── Channels ─────────────────────────────────────────────────────────────────
CH_POS_TX = "pos:transactions"
CH_MATCH_CLOCK = "pos:match_clock"
CH_SIM_CONTROL = "sim:control"

# ── Tunables ─────────────────────────────────────────────────────────────────
SPEED_MULTIPLIER = float(10)          # 10x → 90 min in 9 real min
MATCH_DURATION = 90                   # minutes
TICK_INTERVAL = 1.0                   # real seconds between ticks
BASE_TX_RATE_PER_STALL = 0.35        # avg transactions/stall/sim-second at peak=1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [POS] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pos_simulator")


# ── Demand curve ─────────────────────────────────────────────────────────────

def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def demand_curve(match_minute: float) -> float:
    """Return a 0-1 demand multiplier for the given match minute.

    Three Gaussian peaks:
      - Pre-match  (minute ~5,  σ=8)  — fans arriving, grabbing food
      - Halftime   (minute ~48, σ=5)  — the big rush
      - Late game  (minute ~80, σ=10) — last-chance purchases
    Plus a small baseline so sales never fully stop.
    """
    pre_match = 0.55 * _gauss(match_minute, 5, 8)
    halftime = 1.00 * _gauss(match_minute, 48, 5)
    late_game = 0.40 * _gauss(match_minute, 80, 10)
    baseline = 0.08
    return min(baseline + pre_match + halftime + late_game, 1.0)


# ── Simulator state ─────────────────────────────────────────────────────────

class POSSimulator:
    def __init__(self) -> None:
        self.match_minute: float = 0.0
        self.running: bool = True
        self.total_tx: int = 0
        self.total_revenue: float = 0.0
        self.price_multipliers: dict[str, float] = {
            sid: 1.0 for sid in CONCESSION_STALLS
        }
        self.redis: aioredis.Redis | None = None

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

    # ── Transaction generation ───────────────────────────────────────────────

    def _generate_transaction(self, stall_id: str, demand: float) -> dict | None:
        """Maybe generate a transaction for *stall_id* given current demand."""
        price_mult = self.price_multipliers.get(stall_id, 1.0)
        # Higher prices reduce purchase probability
        price_effect = 1.0 / max(price_mult, 0.1)
        effective_rate = BASE_TX_RATE_PER_STALL * demand * price_effect

        if random.random() > effective_rate:
            return None

        item = random.choice(MENU_ITEMS)
        qty = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        unit_price = round(item["price"] * price_mult, 2)
        total = round(unit_price * qty, 2)

        return {
            "tx_id": uuid.uuid4().hex[:12],
            "timestamp": time.time(),
            "match_minute": round(self.match_minute, 1),
            "stall_id": stall_id,
            "stall_name": CONCESSION_STALLS[stall_id]["name"],
            "item_id": item["id"],
            "item_name": item["name"],
            "qty": qty,
            "unit_price": unit_price,
            "total": total,
            "price_multiplier": price_mult,
        }

    # ── Control listener ─────────────────────────────────────────────────────

    async def _listen_control(self) -> None:
        """Subscribe to sim:control for demo triggers."""
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
            self.total_tx = 0
            self.total_revenue = 0.0
            self.price_multipliers = {sid: 1.0 for sid in CONCESSION_STALLS}
            log.info("🔄 Simulator reset")
        elif action == "set_price":
            stall = payload.get("stall_id")
            mult = float(payload.get("multiplier", 1.0))
            if stall in self.price_multipliers:
                self.price_multipliers[stall] = mult
                log.info("💲 %s price multiplier → %.2f", stall, mult)
            elif stall == "all":
                for sid in self.price_multipliers:
                    self.price_multipliers[sid] = mult
                log.info("💲 ALL stalls price multiplier → %.2f", mult)

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.connect()

        control_task = asyncio.create_task(self._listen_control())
        log.info(
            "POS Simulator started — speed %.0fx, tick %.1fs",
            SPEED_MULTIPLIER, TICK_INTERVAL,
        )

        try:
            while self.running and self.match_minute <= MATCH_DURATION:
                demand = demand_curve(self.match_minute)
                tick_tx = 0
                tick_rev = 0.0

                for stall_id in CONCESSION_STALLS:
                    tx = self._generate_transaction(stall_id, demand)
                    if tx is not None:
                        await self._publish(CH_POS_TX, tx)
                        tick_tx += 1
                        tick_rev += tx["total"]

                self.total_tx += tick_tx
                self.total_revenue += tick_rev

                clock = {
                    "match_minute": round(self.match_minute, 1),
                    "demand": round(demand, 3),
                    "tick_transactions": tick_tx,
                    "tick_revenue": round(tick_rev, 2),
                    "total_transactions": self.total_tx,
                    "total_revenue": round(self.total_revenue, 2),
                    "speed_multiplier": SPEED_MULTIPLIER,
                    "timestamp": time.time(),
                }
                await self._publish(CH_MATCH_CLOCK, clock)

                if int(self.match_minute) % 5 == 0 and self.match_minute == int(self.match_minute):
                    log.info(
                        "Min %2d | demand %.2f | tx %d | rev $%.2f | total tx %d / $%.2f",
                        int(self.match_minute), demand,
                        tick_tx, tick_rev,
                        self.total_tx, self.total_revenue,
                    )

                # Advance match clock
                sim_seconds_per_tick = TICK_INTERVAL * SPEED_MULTIPLIER
                self.match_minute += sim_seconds_per_tick / 60.0

                await asyncio.sleep(TICK_INTERVAL)

        except asyncio.CancelledError:
            log.info("Simulator cancelled")
        finally:
            control_task.cancel()
            try:
                await control_task
            except asyncio.CancelledError:
                pass
            log.info(
                "Match ended — %d transactions, $%.2f revenue",
                self.total_tx, self.total_revenue,
            )
            if self.redis:
                await self.redis.aclose()


# ── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    sim = POSSimulator()
    await sim.run()


if __name__ == "__main__":
    asyncio.run(main())
