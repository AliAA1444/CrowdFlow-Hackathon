from __future__ import annotations

"""CrowdFlow — Safety Agent.

Highest-priority agent (priority 0). Monitors zone density and
occupancy across three escalation tiers:
  - Warning   (>= 3.5 p/m²): general congestion alert
  - Standstill (>= 4.0 p/m²): crowd relief recommended
  - Crush Risk (>= 5.5 p/m²): immediate EMERGENCY_ALERT
Also handles MEDICAL_EMERGENCY override messages from the manual
medical channel.

Run standalone:  python agents/safety_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CONGESTION_HIGH,
    CONGESTION_STATES,
    CRUSH_RISK_DENSITY,
    MEDICAL_OVERRIDE_CHANNEL,
    SAFETY_MAX_DENSITY,
    STANDSTILL_DENSITY,
    get_state_config,
)
from agents.base_agent import BaseAgent, run_agent

# ── Channels ─────────────────────────────────────────────────────────────────
CH_ZONE_METRICS = "broadcast:zone_metrics"

# ── Thresholds ───────────────────────────────────────────────────────────────
DENSITY_CRUSH_RISK  = CRUSH_RISK_DENSITY   # 5.5 p/m² → EMERGENCY_ALERT
DENSITY_STANDSTILL  = STANDSTILL_DENSITY   # 4.0 p/m² → standstill SAFETY_WARNING
DENSITY_WARNING     = CONGESTION_HIGH      # 3.5 p/m² → general SAFETY_WARNING
COUNT_EMERGENCY     = 800                  # absolute count threshold
CLEAR_FACTOR        = 0.80                 # must drop to 80 % of threshold to clear


def _state_visuals(zone: dict) -> tuple[str, str]:
    """Return (congestion_color, congestion_icon) for the zone's current state."""
    key = zone.get("congestion_level") or zone.get("congestion", "free_flow")
    cfg = CONGESTION_STATES.get(key, {})
    return cfg.get("color", "#22c55e"), cfg.get("icon", "🟢")


class SafetyAgent(BaseAgent):
    name = "safety_agent"
    subscribe_channels = [CH_ZONE_METRICS, MEDICAL_OVERRIDE_CHANNEL]

    def __init__(self) -> None:
        super().__init__()
        # zone_id → True while a crush-risk emergency is active
        self.active_emergencies: dict[str, bool] = {}
        # zone_id → True while a standstill warning is active
        self.active_standstill_warnings: dict[str, bool] = {}
        # zone_id → True while a general density warning is active
        self.active_warnings: dict[str, bool] = {}

    async def process(self, channel: str, data: dict) -> None:
        if channel == MEDICAL_OVERRIDE_CHANNEL:
            await self._handle_medical_override(data)
        else:
            await self._handle_zone_metrics(data)

    # ── Medical emergency override ───────────────────────────────────────────

    async def _handle_medical_override(self, data: dict) -> None:
        # Handle clear action from reset endpoint
        if data.get("action") == "clear":
            self.active_emergencies.clear()
            self.active_standstill_warnings.clear()
            self.active_warnings.clear()
            return

        zone_id = data.get("zone_id", "unknown")
        reason  = data.get("reason", "Manual medical emergency override")
        state   = get_state_config("medical_manual")
        await self.publish_decision({
            "action":   "MEDICAL_EMERGENCY",
            "priority": 0,
            "zone_id":  zone_id,
            "details": {
                "reason":           reason,
                "congestion_level": "medical_manual",
                "congestion_color": state["color"],
                "congestion_icon":  state["icon"],
                "timestamp":        data.get("timestamp"),
            },
        })

    # ── Zone metrics handling ────────────────────────────────────────────────

    async def _handle_zone_metrics(self, data: dict) -> None:
        zones = data.get("zones", [])
        for zone in zones:
            zone_id  = zone.get("zone_id")
            density  = zone.get("density", 0.0)
            occupancy = zone.get("occupancy", 0)
            color, icon = _state_visuals(zone)

            is_crush_risk = density >= DENSITY_CRUSH_RISK or occupancy >= COUNT_EMERGENCY
            was_crush_risk = self.active_emergencies.get(zone_id, False)

            # ── Tier 1: Crush Risk → EMERGENCY_ALERT ──────────────────────
            if is_crush_risk and not was_crush_risk:
                self.active_emergencies[zone_id] = True
                # Clear lower tiers when escalating
                self.active_standstill_warnings[zone_id] = False
                self.active_warnings[zone_id] = False
                await self.publish_decision({
                    "action":   "EMERGENCY_ALERT",
                    "priority": 0,
                    "zone_id":  zone_id,
                    "details": {
                        "density":           density,
                        "occupancy":         occupancy,
                        "capacity":          zone.get("capacity"),
                        "threshold_density": DENSITY_CRUSH_RISK,
                        "threshold_count":   COUNT_EMERGENCY,
                        "match_minute":      zone.get("match_minute"),
                        "reason":            self._emergency_reason(density, occupancy),
                        "congestion_level":  zone.get("congestion_level") or zone.get("congestion"),
                        "congestion_color":  color,
                        "congestion_icon":   icon,
                    },
                })

            elif was_crush_risk:
                density_clear = density < DENSITY_CRUSH_RISK * CLEAR_FACTOR
                count_clear   = occupancy < COUNT_EMERGENCY * CLEAR_FACTOR
                if density_clear and count_clear:
                    self.active_emergencies[zone_id] = False
                    await self.publish_decision({
                        "action":   "EMERGENCY_CLEARED",
                        "priority": 0,
                        "zone_id":  zone_id,
                        "details": {
                            "density":      density,
                            "occupancy":    occupancy,
                            "match_minute": zone.get("match_minute"),
                            "congestion_color": color,
                            "congestion_icon":  icon,
                        },
                    })

            # ── Tier 2: Standstill → SAFETY_WARNING (elevated) ────────────
            if not is_crush_risk:
                is_standstill  = density >= DENSITY_STANDSTILL
                was_standstill = self.active_standstill_warnings.get(zone_id, False)

                if is_standstill and not was_standstill:
                    self.active_standstill_warnings[zone_id] = True
                    self.active_warnings[zone_id] = False  # subsume lower tier
                    await self.publish_decision({
                        "action":   "SAFETY_WARNING",
                        "priority": 1,
                        "zone_id":  zone_id,
                        "details": {
                            "density":          density,
                            "occupancy":        occupancy,
                            "congestion":       zone.get("congestion"),
                            "match_minute":     zone.get("match_minute"),
                            "message":          "Zone approaching standstill — prepare crowd relief.",
                            "tier":             "standstill",
                            "congestion_color": color,
                            "congestion_icon":  icon,
                        },
                    })
                elif not is_standstill and was_standstill:
                    self.active_standstill_warnings[zone_id] = False

                # ── Tier 3: General density warning ───────────────────────
                if not is_standstill:
                    is_warning  = density >= DENSITY_WARNING
                    was_warning = self.active_warnings.get(zone_id, False)

                    if is_warning and not was_warning:
                        self.active_warnings[zone_id] = True
                        await self.publish_decision({
                            "action":   "SAFETY_WARNING",
                            "priority": 1,
                            "zone_id":  zone_id,
                            "details": {
                                "density":          density,
                                "occupancy":        occupancy,
                                "congestion":       zone.get("congestion"),
                                "match_minute":     zone.get("match_minute"),
                                "tier":             "warning",
                                "congestion_color": color,
                                "congestion_icon":  icon,
                            },
                        })
                    elif not is_warning and was_warning:
                        self.active_warnings[zone_id] = False

    @staticmethod
    def _emergency_reason(density: float, occupancy: int) -> str:
        reasons = []
        if density >= DENSITY_CRUSH_RISK:
            reasons.append(f"density {density:.1f} p/m² exceeds crush-risk threshold {DENSITY_CRUSH_RISK}")
        if occupancy >= COUNT_EMERGENCY:
            reasons.append(f"count {occupancy} exceeds {COUNT_EMERGENCY}")
        return "; ".join(reasons)


if __name__ == "__main__":
    run_agent(SafetyAgent)
