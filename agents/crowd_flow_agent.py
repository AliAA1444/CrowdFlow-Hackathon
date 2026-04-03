from __future__ import annotations

"""CrowdFlow — Crowd Flow Agent.

Monitors zone metrics, maintains a rolling window of occupancy counts,
detects growth trends, and emits REROUTE_TRAFFIC decisions when a zone
is filling dangerously fast.

When all interior zones are at "congested" or above, suggests rerouting
overflow to exterior activation zones.

Run standalone:  python agents/crowd_flow_agent.py
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CONGESTION_STATES,
    SLOW_FLOW_DENSITY,
    ZONES,
    get_state_config,
)
from core.schemas import ZoneUpdate
from agents.base_agent import AgentAction, BaseAgent, run_agent

# ── Channels ─────────────────────────────────────────────────────────────────
CH_ZONE_METRICS  = "broadcast:zone_metrics"
CH_ZONE_UPDATES  = "zone_updates"

# ── Tunables ─────────────────────────────────────────────────────────────────
ROLLING_WINDOW    = 30    # keep last N readings per zone
RECENT_WINDOW     = 5     # compare most-recent N …
OLDER_WINDOW      = 5     # … against the N before them
GROWTH_THRESHOLD  = 25.0  # % growth that triggers reroute
PREDICTION_MINUTES = 5.0  # minutes ahead for linear prediction

# State severity ordering (used to decide overflow)
_STATE_SEVERITY: dict[str, int] = {
    "free_flow":     0,
    "moderate":      1,
    "slow_flow":     2,
    "congested":     3,
    "standstill":    4,
    "crush_risk":    5,
    "medical_manual": 6,
}
_OVERFLOW_THRESHOLD = _STATE_SEVERITY["congested"]   # 3 — congested or above

# Exterior zones available as overflow targets (ordered by preference)
_OVERFLOW_ZONES = ["zone_activation_1", "zone_activation_2"]

# Interior zone ids (resolved once at import time)
_INTERIOR_ZONES = [zid for zid, cfg in ZONES.items() if cfg.get("zone_type") == "interior"]

# Adjacency map for flow-based routing recommendations
_ZONE_ADJACENCY: dict[str, list[str]] = {
    "zone_north_gate":      ["zone_north_concourse", "zone_west_stand", "zone_east_stand", "zone_activation_1"],
    "zone_south_gate":      ["zone_south_concourse", "zone_west_stand", "zone_east_stand", "zone_activation_2"],
    "zone_west_stand":      ["zone_north_gate", "zone_south_gate", "zone_north_concourse", "zone_south_concourse", "zone_pitch_view"],
    "zone_east_stand":      ["zone_north_gate", "zone_south_gate", "zone_north_concourse", "zone_south_concourse", "zone_activation_1", "zone_activation_2"],
    "zone_north_concourse": ["zone_north_gate", "zone_west_stand", "zone_east_stand", "zone_pitch_view"],
    "zone_south_concourse": ["zone_south_gate", "zone_west_stand", "zone_east_stand", "zone_pitch_view"],
    "zone_pitch_view":      ["zone_north_concourse", "zone_south_concourse", "zone_west_stand"],
    "zone_activation_1":    ["zone_north_gate", "zone_east_stand", "zone_north_parking"],
    "zone_activation_2":    ["zone_south_gate", "zone_east_stand", "zone_north_parking"],
    "zone_north_parking":   ["zone_activation_1", "zone_activation_2"],
}


class CrowdFlowAgent(BaseAgent):
    name = "crowd_flow_agent"
    subscribe_channels = [CH_ZONE_METRICS, CH_ZONE_UPDATES]

    def __init__(self) -> None:
        super().__init__()
        # zone_id → deque of occupancy counts (for legacy zone_metrics handler)
        self.history: dict[str, collections.deque] = {
            zid: collections.deque(maxlen=ROLLING_WINDOW)
            for zid in ZONES
        }
        # zone_id → latest congestion state key
        self.zone_congestion: dict[str, str] = {}
        # zone_id → latest ZoneUpdate (for cross-zone density comparison)
        self._zone_updates: dict[str, ZoneUpdate] = {}

    async def process(self, channel: str, data: dict) -> None:
        if channel == CH_ZONE_UPDATES:
            update = ZoneUpdate(**data)
            actions = await self.evaluate(update)
            await self.publish_actions(actions)
            return

        zones = data.get("zones", [])

        # Update congestion state cache for all reported zones
        for zone in zones:
            zid = zone.get("zone_id")
            if zid:
                key = zone.get("congestion_level") or zone.get("congestion", "free_flow")
                self.zone_congestion[zid] = key

        for zone in zones:
            zone_id = zone.get("zone_id")
            if zone_id not in self.history:
                continue

            occupancy = zone.get("occupancy", 0)
            self.history[zone_id].append(occupancy)

            growth = self._compute_growth(zone_id)
            if growth is None:
                continue

            density = zone.get("density", 0.0)
            if growth >= GROWTH_THRESHOLD and density >= SLOW_FLOW_DENSITY:
                recent_avg = self._avg_recent(zone_id)
                prediction = self._predict(zone_id)
                congestion_key = self.zone_congestion.get(zone_id, "free_flow")
                state = get_state_config(congestion_key)
                target_zone = self._pick_overflow_target()
                await self.publish_decision({
                    "action":   "REROUTE_TRAFFIC",
                    "priority": 2,
                    "zone_id":  zone_id,
                    "details": {
                        "current_count":        occupancy,
                        "growth_rate_pct":      round(growth, 1),
                        "recent_avg":           round(recent_avg, 1),
                        "predicted_count_5min": round(prediction, 0),
                        "density":              zone.get("density"),
                        "congestion":           zone.get("congestion"),
                        "match_minute":         zone.get("match_minute"),
                        "suggested_target_zone": target_zone,
                        "congestion_color":     state["color"],
                        "congestion_icon":      state["icon"],
                    },
                })

    # ── ZoneUpdate evaluation (Prompt 5) ────────────────────────────────────

    async def evaluate(self, update: ZoneUpdate) -> list[AgentAction]:
        """Flow-based routing and gate capacity recommendations from ZoneUpdate."""
        guard = await super().evaluate(update)
        if update.source == "stale_fallback":
            return guard

        # Keep a live snapshot of all zones for cross-zone density comparison
        self._zone_updates[update.zone_id] = update

        actions: list[AgentAction] = []
        zid = update.zone_id

        # ── Flow-based rerouting ──────────────────────────────────────────
        if update.density > 3.0:
            adjacent = _ZONE_ADJACENCY.get(zid, [])
            for adj_id in adjacent:
                adj = self._zone_updates.get(adj_id)
                if adj and adj.density < 1.5:
                    # Prefer a target whose direction aligns with current flow
                    # (flow_direction is degrees clockwise-from-east in image space)
                    actions.append(AgentAction(
                        action="REROUTE_TRAFFIC",
                        zone_id=zid,
                        priority="HIGH",
                        detail=(
                            f"Dense zone {zid} ({update.density:.1f} p/m²) → "
                            f"reroute to {adj_id} ({adj.density:.1f} p/m²); "
                            f"flow direction {update.flow_direction:.0f}°"
                        ),
                    ))
                    break  # One recommendation per evaluation cycle

        # ── Gate capacity check ───────────────────────────────────────────
        if "gate" in zid and (update.gate_in > 0 or update.gate_out > 0):
            net = update.gate_in - update.gate_out
            capacity = ZONES.get(zid, {}).get("capacity", 500)
            if net > capacity * 0.8:
                actions.append(AgentAction(
                    action="CLOSE_GATE",
                    zone_id=zid,
                    priority="HIGH",
                    detail=(
                        f"Gate {zid}: net occupancy {net} exceeds 80% of "
                        f"capacity {capacity} (IN={update.gate_in}, OUT={update.gate_out}). "
                        "Recommend closing to new entry."
                    ),
                ))

        return actions

    def _pick_overflow_target(self) -> str | None:
        """
        Return an exterior overflow zone if all interior zones are congested or above.
        Otherwise return None (normal reroute within interior).
        """
        interior_states = [
            self.zone_congestion.get(zid, "free_flow")
            for zid in _INTERIOR_ZONES
        ]
        all_interior_congested = all(
            _STATE_SEVERITY.get(s, 0) >= _OVERFLOW_THRESHOLD
            for s in interior_states
        )

        if all_interior_congested:
            # Pick the overflow zone with the lowest current severity
            best = min(
                _OVERFLOW_ZONES,
                key=lambda zid: _STATE_SEVERITY.get(
                    self.zone_congestion.get(zid, "free_flow"), 0
                ),
            )
            return best

        return None

    # ── Analytics ────────────────────────────────────────────────────────────

    def _compute_growth(self, zone_id: str) -> float | None:
        """Return % growth of recent vs older window, or None if insufficient data."""
        buf = self.history[zone_id]
        needed = RECENT_WINDOW + OLDER_WINDOW
        if len(buf) < needed:
            return None

        items = list(buf)
        older  = items[-(RECENT_WINDOW + OLDER_WINDOW):-RECENT_WINDOW]
        recent = items[-RECENT_WINDOW:]

        avg_older  = sum(older) / len(older)
        avg_recent = sum(recent) / len(recent)

        if avg_older <= 0:
            return None
        return ((avg_recent - avg_older) / avg_older) * 100.0

    def _avg_recent(self, zone_id: str) -> float:
        buf = self.history[zone_id]
        recent = list(buf)[-RECENT_WINDOW:]
        return sum(recent) / len(recent) if recent else 0.0

    def _predict(self, zone_id: str) -> float:
        """Simple linear extrapolation: project the recent growth forward."""
        buf = self.history[zone_id]
        needed = RECENT_WINDOW + OLDER_WINDOW
        if len(buf) < needed:
            return list(buf)[-1] if buf else 0.0

        items = list(buf)
        older_avg  = sum(items[-(RECENT_WINDOW + OLDER_WINDOW):-RECENT_WINDOW]) / OLDER_WINDOW
        recent_avg = sum(items[-RECENT_WINDOW:]) / RECENT_WINDOW

        # Each reading ≈ 2 s real time. Growth per reading:
        growth_per_reading = (recent_avg - older_avg) / OLDER_WINDOW
        # Readings in 5 sim-minutes (10× speed, 2 s interval → ~15 readings per sim-min)
        readings_ahead = (PREDICTION_MINUTES * 60) / 2.0
        return recent_avg + growth_per_reading * readings_ahead


if __name__ == "__main__":
    run_agent(CrowdFlowAgent)
