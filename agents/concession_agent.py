from __future__ import annotations

"""CrowdFlow — Concession Agent.

Revenue-optimisation agent (lowest priority). Monitors POS velocity,
zone traffic, and fan intent signals. Recommends dynamic pricing and
flash deals subject to orchestrator approval.

Now exterior-zone aware: if zone_activation_1 or zone_activation_2 are
lightly occupied, the agent suggests flash deals on nearby stalls to draw
foot traffic toward those zones and relieve interior congestion.

Run standalone:  python agents/concession_agent.py
"""

import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CONCESSION_STALLS,
    CONGESTION_STATES,
    ZONES,
    get_state_config,
)
from agents.base_agent import BaseAgent, run_agent

# ── Channels ─────────────────────────────────────────────────────────────────
CH_POS_TX      = "pos:transactions"
CH_ZONE_METRICS = "broadcast:zone_metrics"
CH_FAN_INTENT  = "fan:food_intent"

# ── Tunables ─────────────────────────────────────────────────────────────────
HIGH_VELOCITY_THRESHOLD  = 40     # txns per minute triggers surge pricing
VELOCITY_WINDOW          = 60.0   # seconds to measure tx velocity
LOW_TRAFFIC_UTILIZATION  = 0.25   # zone utilization below this = low traffic
FLASH_DEAL_COOLDOWN      = 120.0  # seconds between flash deals for same zone
INTENT_THRESHOLD         = 15     # food intents needed to trigger promotion
INTENT_WINDOW            = 60.0   # seconds for intent accumulation

# Exterior zones that can act as overflow attractors
_EXTERIOR_OVERFLOW_ZONES = ["zone_activation_1", "zone_activation_2"]


class ConcessionAgent(BaseAgent):
    name = "concession_agent"
    subscribe_channels = [CH_POS_TX, CH_ZONE_METRICS, CH_FAN_INTENT]

    def __init__(self) -> None:
        super().__init__()
        # stall_id → deque of transaction timestamps
        self.tx_timestamps: dict[str, collections.deque] = {
            sid: collections.deque() for sid in CONCESSION_STALLS
        }
        # zone_id → latest zone metric
        self.zone_state: dict[str, dict] = {}
        # zone_id → last flash deal timestamp (for cooldown)
        self.last_flash_deal: dict[str, float] = {}
        # zone_id → deque of intent timestamps
        self.intent_timestamps: dict[str, collections.deque] = {
            zid: collections.deque() for zid in ZONES
        }
        # stall_id → True if we already recommended surge (avoid spam)
        self.surge_active: dict[str, bool] = {
            sid: False for sid in CONCESSION_STALLS
        }

    async def process(self, channel: str, data: dict) -> None:
        if channel == CH_POS_TX:
            await self._handle_transaction(data)
        elif channel == CH_ZONE_METRICS:
            await self._handle_zone_metrics(data)
        elif channel == CH_FAN_INTENT:
            await self._handle_food_intent(data)

    # ── POS transaction handling ─────────────────────────────────────────────

    async def _handle_transaction(self, tx: dict) -> None:
        stall_id = tx.get("stall_id")
        if stall_id not in self.tx_timestamps:
            return

        now = time.time()
        self.tx_timestamps[stall_id].append(now)

        # Prune old timestamps
        cutoff = now - VELOCITY_WINDOW
        while self.tx_timestamps[stall_id] and self.tx_timestamps[stall_id][0] < cutoff:
            self.tx_timestamps[stall_id].popleft()

        velocity = len(self.tx_timestamps[stall_id]) * (60.0 / VELOCITY_WINDOW)

        if velocity >= HIGH_VELOCITY_THRESHOLD and not self.surge_active[stall_id]:
            self.surge_active[stall_id] = True
            stall_cfg = CONCESSION_STALLS[stall_id]
            zone_id   = stall_cfg["zone"]
            color, icon = self._zone_visuals(zone_id)
            await self.publish_decision({
                "action":   "DYNAMIC_PRICE_UP",
                "priority": 3,
                "zone_id":  zone_id,
                "details": {
                    "stall_id":               stall_id,
                    "stall_name":             stall_cfg["name"],
                    "velocity_per_min":       round(velocity, 1),
                    "recommended_multiplier": 1.15,
                    "match_minute":           tx.get("match_minute"),
                    "congestion_color":       color,
                },
            })
        elif velocity < HIGH_VELOCITY_THRESHOLD * 0.7:
            # Reset surge flag when velocity drops
            self.surge_active[stall_id] = False

    # ── Zone metrics handling ────────────────────────────────────────────────

    async def _handle_zone_metrics(self, data: dict) -> None:
        zones = data.get("zones", [])
        for zone in zones:
            zone_id = zone.get("zone_id")
            self.zone_state[zone_id] = zone

        await self._check_flash_deals(data.get("match_minute"))
        await self._check_exterior_overflow_deals(data.get("match_minute"))

    async def _check_flash_deals(self, match_minute: float | None) -> None:
        """Find low-traffic interior zones that have stalls and recommend flash deals."""
        now = time.time()

        for stall_id, stall_cfg in CONCESSION_STALLS.items():
            zone_id = stall_cfg["zone"]
            zone    = self.zone_state.get(zone_id)
            if zone is None:
                continue

            utilization = zone.get("utilization", 1.0)
            if utilization >= LOW_TRAFFIC_UTILIZATION:
                continue

            last = self.last_flash_deal.get(zone_id, 0.0)
            if now - last < FLASH_DEAL_COOLDOWN:
                continue

            color, icon = self._zone_visuals(zone_id)
            self.last_flash_deal[zone_id] = now
            await self.publish_decision({
                "action":   "FLASH_DEAL",
                "priority": 3,
                "zone_id":  zone_id,
                "details": {
                    "stall_id":             stall_id,
                    "stall_name":           stall_cfg["name"],
                    "zone_utilization":     round(utilization, 3),
                    "recommended_discount": 0.20,
                    "match_minute":         match_minute,
                    "congestion_color":     color,
                },
            })

    async def _check_exterior_overflow_deals(self, match_minute: float | None) -> None:
        """
        If an exterior activation zone has low density, push a flash deal on a
        nearby stall to draw foot traffic there and relieve interior pressure.
        """
        now = time.time()

        for ext_zone_id in _EXTERIOR_OVERFLOW_ZONES:
            zone = self.zone_state.get(ext_zone_id)
            if zone is None:
                continue

            utilization = zone.get("utilization", 1.0)
            if utilization >= LOW_TRAFFIC_UTILIZATION:
                continue

            last = self.last_flash_deal.get(ext_zone_id, 0.0)
            if now - last < FLASH_DEAL_COOLDOWN:
                continue

            # Pick the stall with the lowest current velocity (least busy)
            stall_id, stall_cfg = self._least_busy_stall()
            color, icon = self._zone_visuals(ext_zone_id)
            self.last_flash_deal[ext_zone_id] = now
            await self.publish_decision({
                "action":   "FLASH_DEAL",
                "priority": 3,
                "zone_id":  ext_zone_id,
                "details": {
                    "stall_id":             stall_id,
                    "stall_name":           stall_cfg["name"],
                    "zone_utilization":     round(utilization, 3),
                    "recommended_discount": 0.25,   # extra incentive to move outdoors
                    "match_minute":         match_minute,
                    "reason":               f"Draw traffic to {zone.get('label', ext_zone_id)} (exterior overflow)",
                    "congestion_color":     color,
                },
            })

    # ── Fan food-intent handling ─────────────────────────────────────────────

    async def _handle_food_intent(self, data: dict) -> None:
        zone_id = data.get("zone_id")
        if zone_id not in self.intent_timestamps:
            return

        now = time.time()
        self.intent_timestamps[zone_id].append(now)

        # Prune old intents
        cutoff = now - INTENT_WINDOW
        while self.intent_timestamps[zone_id] and self.intent_timestamps[zone_id][0] < cutoff:
            self.intent_timestamps[zone_id].popleft()

        count = len(self.intent_timestamps[zone_id])
        if count >= INTENT_THRESHOLD:
            stall_id, stall_cfg = self._nearest_stall(zone_id)
            await self.publish_decision({
                "action":   "TARGETED_PROMOTION",
                "priority": 3,
                "zone_id":  zone_id,
                "details": {
                    "intent_count":     count,
                    "window_seconds":   INTENT_WINDOW,
                    "target_stall_id":  stall_id,
                    "target_stall_name": stall_cfg["name"],
                    "match_minute":     data.get("match_minute"),
                },
            })
            self.intent_timestamps[zone_id].clear()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _zone_visuals(self, zone_id: str) -> tuple[str, str]:
        """Return (congestion_color, congestion_icon) for a zone."""
        zone = self.zone_state.get(zone_id, {})
        key  = zone.get("congestion_level") or zone.get("congestion", "free_flow")
        cfg  = CONGESTION_STATES.get(key, {})
        return cfg.get("color", "#22c55e"), cfg.get("icon", "🟢")

    def _least_busy_stall(self) -> tuple[str, dict]:
        """Return the stall with the fewest recent transactions."""
        return min(
            CONCESSION_STALLS.items(),
            key=lambda kv: len(self.tx_timestamps.get(kv[0], [])),
        )

    @staticmethod
    def _nearest_stall(zone_id: str) -> tuple[str, dict]:
        """Return the stall in or closest to *zone_id*."""
        for sid, cfg in CONCESSION_STALLS.items():
            if cfg["zone"] == zone_id:
                return sid, cfg
        sid = next(iter(CONCESSION_STALLS))
        return sid, CONCESSION_STALLS[sid]


if __name__ == "__main__":
    run_agent(ConcessionAgent)
