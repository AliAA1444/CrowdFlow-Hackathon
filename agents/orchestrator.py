from __future__ import annotations

"""CrowdFlow — Orchestrator.

The central conflict-resolution engine. Collects agent decisions,
resolves conflicts every 2 seconds using the priority hierarchy
Safety > Crowd Flow > Revenue, and publishes approved/denied actions
to the dashboard.

Updated conflict rules (granular states):
  - DYNAMIC_PRICE_UP / FLASH_DEAL denied at "standstill" or above.
  - REROUTE_TRAFFIC with suggested_target_zone denied if target is at
    "slow_flow" or above.
  - MEDICAL_EMERGENCY treated as priority-0 override (same as EMERGENCY_ALERT).

Run standalone:  python agents/orchestrator.py
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import redis.asyncio as aioredis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CONGESTION_HIGH,
    CONGESTION_MODERATE,
    CONGESTION_STATES,
    REDIS_URL,
    ZONES,
    classify_congestion,
    get_state_config,
)
from agents.base_agent import BaseAgent, CH_ORCHESTRATOR

# ── Channels ─────────────────────────────────────────────────────────────────
CH_DASHBOARD_ACTIONS = "dashboard:actions"

# ── Tunables ─────────────────────────────────────────────────────────────────
RESOLVE_INTERVAL = 2.0   # seconds between resolution sweeps

# Redis stream for audit history
STREAM_KEY = "orchestrator:history"

# State severity — used for threshold comparisons in conflict rules
STATE_SEVERITY: dict[str, int] = {
    "free_flow":      0,
    "moderate":       1,
    "slow_flow":      2,
    "congested":      3,
    "standstill":     4,
    "crush_risk":     5,
    "medical_manual": 6,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORCH] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")


class Orchestrator(BaseAgent):
    name = "orchestrator"
    subscribe_channels = [CH_ORCHESTRATOR]

    def __init__(self) -> None:
        super().__init__()
        self.decision_buffer: list[dict] = []
        self._resolver_task: asyncio.Task | None = None
        # zone_id → True while an emergency is active in that zone
        self.active_emergencies: dict[str, bool] = {}
        # zone_id → latest congestion state key
        self.zone_congestion: dict[str, str] = {}

    # ── Lifecycle override ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the resolver loop alongside the normal message listener."""
        await self.connect()

        # Subscribe to zone metrics too (for congestion state), on a second pubsub
        self._zone_sub = self.redis.pubsub()
        try:
            await self._zone_sub.subscribe("broadcast:zone_metrics")
        except Exception as exc:
            log.error("Could not subscribe to zone metrics: %s", exc)

        self._resolver_task = asyncio.create_task(self._resolve_loop())
        self._zone_task     = asyncio.create_task(self._zone_listener())

        # Now start normal BaseAgent listener
        self._pubsub = self.redis.pubsub()
        try:
            await self._pubsub.subscribe(*self.subscribe_channels)
            log.info("Orchestrator listening on %s", CH_ORCHESTRATOR)
        except Exception as exc:
            log.error("Subscribe failed: %s", exc)
            raise

        try:
            async for msg in self._pubsub.listen():
                if not self._running:
                    break
                if msg["type"] != "message":
                    continue
                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    continue
                try:
                    await self.process(msg["channel"], data)
                except Exception as exc:
                    log.error("Error processing: %s", exc, exc_info=True)
        except asyncio.CancelledError:
            pass
        finally:
            if self._resolver_task:
                self._resolver_task.cancel()
            if self._zone_task:
                self._zone_task.cancel()
            await self.shutdown()

    async def _zone_listener(self) -> None:
        """Track zone congestion levels for conflict rules."""
        try:
            async for msg in self._zone_sub.listen():
                if msg["type"] != "message":
                    continue
                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    continue
                for zone in data.get("zones", []):
                    zid = zone.get("zone_id")
                    if zid:
                        # Accept granular state key or fall back to legacy field
                        key = zone.get("congestion_level") or zone.get("congestion", "free_flow")
                        self.zone_congestion[zid] = key
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("Zone listener error: %s", exc)

    # ── Message intake ───────────────────────────────────────────────────────

    async def process(self, channel: str, data: dict) -> None:
        """Buffer incoming decisions for the next resolve cycle."""
        self.decision_buffer.append(data)

    # ── Resolution loop ──────────────────────────────────────────────────────

    async def _resolve_loop(self) -> None:
        log.info("Resolver loop started (every %.1fs)", RESOLVE_INTERVAL)
        try:
            while self._running:
                await asyncio.sleep(RESOLVE_INTERVAL)
                if self.decision_buffer:
                    await self._resolve()
        except asyncio.CancelledError:
            pass

    async def _resolve(self) -> None:
        """Process buffered decisions: sort by priority, apply conflict rules."""
        batch = self.decision_buffer[:]
        self.decision_buffer.clear()

        # Sort by priority (0 = highest)
        batch.sort(key=lambda d: d.get("priority", 99))

        for decision in batch:
            action   = decision.get("action", "")
            zone_id  = decision.get("zone_id")
            priority = decision.get("priority", 99)

            status, note = self._evaluate(action, zone_id, priority, decision)

            # Track emergency state
            if action in ("EMERGENCY_ALERT", "MEDICAL_EMERGENCY") and status == "approved":
                self.active_emergencies[zone_id] = True
            elif action == "EMERGENCY_CLEARED" and status == "approved":
                self.active_emergencies.pop(zone_id, None)

            # Attach zone visuals to the dispatched result
            congestion_key = self.zone_congestion.get(zone_id, "free_flow") if zone_id else "free_flow"
            state_cfg      = CONGESTION_STATES.get(congestion_key, {})

            result = {
                "agent":              decision.get("agent"),
                "action":             action,
                "priority":           priority,
                "zone_id":            zone_id,
                "status":             status,
                "orchestrator_note":  note,
                "details":            decision.get("details", {}),
                "timestamp":          time.time(),
                "congestion_color":   state_cfg.get("color", "#22c55e"),
                "congestion_icon":    state_cfg.get("icon",  "🟢"),
            }

            # Publish to dashboard
            await self._publish_action(result)
            # Append to Redis stream for history
            await self._log_to_stream(result)

            emoji = "✅" if status == "approved" else "❌"
            log.info(
                "%s %s %s from %s — %s",
                emoji, status.upper(), action,
                decision.get("agent"), note,
            )

    def _evaluate(
        self, action: str, zone_id: str | None, priority: int, decision: dict,
    ) -> tuple[str, str]:
        """Apply conflict rules. Returns (status, note)."""

        # ── Rule 1: Emergency and medical always approved ─────────────────
        if action in ("EMERGENCY_ALERT", "MEDICAL_EMERGENCY"):
            return "approved", "EMERGENCY: immediate override — all non-safety actions suspended"

        if action == "EMERGENCY_CLEARED":
            return "approved", "Emergency cleared — normal operations resumed"

        # ── Rule 2: During active emergency, deny all non-safety actions ──
        if zone_id and self.active_emergencies.get(zone_id):
            if priority > 0:
                return (
                    "denied",
                    f"Active emergency in {zone_id} — all non-safety actions blocked",
                )

        # ── Rule 3: Safety warnings always approved ───────────────────────
        if action == "SAFETY_WARNING":
            return "approved", "Safety warning acknowledged"

        # Resolve current zone severity (works for both old and new state keys)
        zone_severity = STATE_SEVERITY.get(
            self.zone_congestion.get(zone_id, "free_flow") if zone_id else "free_flow",
            0,
        )

        # ── Rule 4: Deny surge pricing at standstill or above ─────────────
        if action == "DYNAMIC_PRICE_UP" and zone_id:
            if zone_severity >= STATE_SEVERITY["standstill"]:
                return (
                    "denied",
                    "Price increase blocked — zone at standstill or above (safety priority)",
                )
            if zone_severity >= STATE_SEVERITY["congested"]:
                return (
                    "denied",
                    "Price increase in congested zone traps people, conflicting with crowd safety",
                )
            if zone_severity >= STATE_SEVERITY["moderate"]:
                return (
                    "denied",
                    "Price increase in moderately congested zone risks worsening crowd flow",
                )

        # ── Rule 5: Deny flash deals that attract people to congested zones
        if action == "FLASH_DEAL" and zone_id:
            if zone_severity >= STATE_SEVERITY["standstill"]:
                return (
                    "denied",
                    "Flash deal blocked — target zone at standstill or above",
                )
            if zone_severity >= STATE_SEVERITY["congested"]:
                return (
                    "denied",
                    "Flash deal would attract more people to already congested zone",
                )

        # ── Rule 6: Reroute traffic — verify suggested target is clear ────
        if action == "REROUTE_TRAFFIC":
            target_zone = decision.get("details", {}).get("suggested_target_zone")
            if target_zone:
                target_severity = STATE_SEVERITY.get(
                    self.zone_congestion.get(target_zone, "free_flow"), 0
                )
                if target_severity >= STATE_SEVERITY["slow_flow"]:
                    return (
                        "denied",
                        f"Target zone '{target_zone}' is also congested — reroute denied",
                    )
            return "approved", "Reroute approved — crowd flow optimisation"

        # ── Rule 7: Targeted promotion (low priority, usually OK) ─────────
        if action == "TARGETED_PROMOTION" and zone_id:
            if zone_severity >= STATE_SEVERITY["congested"]:
                return (
                    "denied",
                    "Cannot run promotion in congested zone — safety risk",
                )

        # ── Default: approve ──────────────────────────────────────────────
        return "approved", "Action approved"

    # ── Publishing helpers ───────────────────────────────────────────────────

    async def _publish_action(self, result: dict) -> None:
        try:
            await self.redis.publish(CH_DASHBOARD_ACTIONS, json.dumps(result))
        except Exception as exc:
            log.error("Publish to %s failed: %s", CH_DASHBOARD_ACTIONS, exc)

    async def _log_to_stream(self, result: dict) -> None:
        try:
            await self.redis.xadd(
                STREAM_KEY,
                {"data": json.dumps(result)},
                maxlen=1000,
            )
        except Exception as exc:
            log.error("Stream append failed: %s", exc)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    orch = Orchestrator()
    asyncio.run(orch.start())
