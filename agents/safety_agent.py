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
from core.schemas import ZoneUpdate
from core.zone_state_cache import ZoneStateCache
from agents.base_agent import AgentAction, BaseAgent, run_agent

# ── Channels ─────────────────────────────────────────────────────────────────
CH_ZONE_METRICS  = "broadcast:zone_metrics"
CH_ZONE_UPDATES  = "zone_updates"

# ── Thresholds (legacy, used by _handle_zone_metrics for broadcast:zone_metrics) ─
DENSITY_CRUSH_RISK  = CRUSH_RISK_DENSITY   # 5.5 p/m²
DENSITY_STANDSTILL  = STANDSTILL_DENSITY   # 4.0 p/m²
DENSITY_WARNING     = CONGESTION_HIGH      # 3.5 p/m²
COUNT_EMERGENCY     = 800
CLEAR_FACTOR        = 0.80

# ── Small-venue thresholds for evaluate() ────────────────────────────────────
# Zones range 40–1200 sqm; with imgsz=1280/conf=0.15 CV tuning, typical live
# counts rise to ~60–200 people; density range in practice 0.05–0.55 p/sqm.

# Threshold 0.45: ~180 people in a 400 sqm concourse — genuine high density
# for a small venue; previously 0.35 but doubled detection doubles raw density
SMALL_DENSITY_CRITICAL = 0.45
# De-escalate at ~82% of critical (0.08 hysteresis gap prevents oscillation)
SMALL_DENSITY_CRITICAL_CLEAR = 0.37

# Threshold 0.28: ~112 people in 400 sqm — elevated but not critical
# (previously 0.20; scaled up proportionally with detection improvement)
SMALL_DENSITY_WARNING = 0.28
# De-escalate at ~64% of warning threshold (0.10 hysteresis gap)
SMALL_DENSITY_WARNING_CLEAR = 0.18

# Threshold 4.0 px/frame: fast movement — at the top of observed range
# (previously 3.5; bumped to stay above normal busy-crowd flow after CV fix)
SMALL_FLOW_SURGE = 4.0
# Minimum density to flag a surge as meaningful (not an empty-zone artefact)
SMALL_FLOW_SURGE_MIN_DENSITY = 0.20

# Threshold 0.25: cross-zone overload check — zone AND all neighbours busy
SMALL_DENSITY_OVERLOAD = 0.25
# Minimum neighbour density to count as "under pressure"
# (previously 0.15; kept — adjacent zones may not have improved CV uniformly)
SMALL_ADJACENT_PRESSURE = 0.15

# Stagnation: near-zero flow in a dense zone signals a bottleneck
# Threshold 0.30: ~120 people in 400 sqm — crowd too dense to be static safely
SMALL_STAGNATION_MIN_DENSITY = 0.30


_ZONE_AR: dict[str, str] = {
    "gate_main":      "البوابة الرئيسية",
    "gate_vip":       "بوابة VIP",
    "gate_south":     "البوابة الجنوبية",
    "concourse_main": "الردهة الرئيسية",
    "stands_east":    "المدرج الشرقي",
    "stands_west":    "المدرج الغربي",
    "stands_north":   "المدرج الشمالي",
    "food_court":     "منطقة الطعام",
    "activation_zone": "منطقة المشجعين",
}


def _zone_ar(zone_id: str) -> str:
    return _ZONE_AR.get(zone_id, zone_id)


def _state_visuals(zone: dict) -> tuple[str, str]:
    """Return (congestion_color, congestion_icon) for the zone's current state."""
    key = zone.get("congestion_level") or zone.get("congestion", "free_flow")
    cfg = CONGESTION_STATES.get(key, {})
    return cfg.get("color", "#22c55e"), cfg.get("icon", "🟢")


class SafetyAgent(BaseAgent):
    name = "safety_agent"
    subscribe_channels = [CH_ZONE_METRICS, MEDICAL_OVERRIDE_CHANNEL, CH_ZONE_UPDATES]

    def __init__(self, zone_state_cache: ZoneStateCache | None = None) -> None:
        super().__init__()
        self._zone_cache = zone_state_cache
        # zone_id → True while a crush-risk emergency is active
        self.active_emergencies: dict[str, bool] = {}
        # zone_id → True while a standstill warning is active
        self.active_standstill_warnings: dict[str, bool] = {}
        # zone_id → True while a general density warning is active
        self.active_warnings: dict[str, bool] = {}

    async def process(self, channel: str, data: dict) -> None:
        if channel == MEDICAL_OVERRIDE_CHANNEL:
            await self._handle_medical_override(data)
        elif channel == CH_ZONE_UPDATES:
            update = ZoneUpdate(**data)
            actions = await self.evaluate(update)
            await self.publish_actions(actions)
        else:
            await self._handle_zone_metrics(data)

    # ── ZoneUpdate evaluation ────────────────────────────────────────────────

    async def evaluate(self, update: ZoneUpdate) -> list[AgentAction]:
        """Evaluate density + flow rules from a live ZoneUpdate.

        Thresholds are calibrated for small-venue footage (zones 40–1200 sqm,
        typical counts 30–150 people, density range 0.05–0.40 p/sqm).

        Must call super() first — base class skips evaluation on stale data.
        """
        guard = await super().evaluate(update)
        if update.source == "stale_fallback":
            return guard

        actions: list[AgentAction] = []
        zid = update.zone_id
        d   = update.density
        mag = update.flow_magnitude

        # ── Density rules (small-venue calibrated, imgsz=1280 CV baseline) ─
        # 0.45 p/sqm = ~180 people in 400 sqm — zone at maximum safe capacity
        if d > SMALL_DENSITY_CRITICAL:
            actions.append(AgentAction(
                action="CRITICAL: Zone at maximum safe capacity",
                zone_id=zid,
                priority="critical",
                detail=(
                    f"كثافة {_zone_ar(zid)} ({d:.3f} شخص/م²) تتجاوز الحد الحرج "
                    f"{SMALL_DENSITY_CRITICAL} شخص/م² "
                    f"(ما يعادل ~{int(d * 400)} شخصاً). "
                    f"مطلوب تدخل فوري لتخفيف الحشد."
                ),
                public_message=(
                    f"🚨 تنبيه عاجل: {_zone_ar(zid)} مكتظ بشكل كامل. "
                    f"يُرجى الانتقال فوراً إلى منطقة مجاورة لسلامتكم."
                ),
            ))
        elif d > SMALL_DENSITY_WARNING:
            # 0.28 p/sqm = ~112 people in 400 sqm — elevated but not critical
            actions.append(AgentAction(
                action="WARNING: Zone approaching capacity limit",
                zone_id=zid,
                priority="high",
                detail=(
                    f"كثافة {_zone_ar(zid)} ({d:.3f} شخص/م²) تتجاوز حد التحذير "
                    f"{SMALL_DENSITY_WARNING} شخص/م² "
                    f"(ما يعادل ~{int(d * 400)} شخصاً). "
                    f"المراقبة مستمرة — الاستعداد لتوجيه الحشد."
                ),
                public_message=(
                    f"⚠️ {_zone_ar(zid)} يشهد ازدحاماً متزايداً. "
                    f"يُنصح بتجنب هذه المنطقة مؤقتاً واختيار بديل أكثر راحة."
                ),
            ))

        # ── Flow rules (calibrated to 0.5–4.0 px/frame observed range) ───
        # 3.5 px/frame: near top of observed range, indicating rapid movement
        if mag > SMALL_FLOW_SURGE and d > SMALL_FLOW_SURGE_MIN_DENSITY:
            actions.append(AgentAction(
                action="ALERT: Rapid crowd surge detected — abnormal movement speed in occupied zone",
                zone_id=zid,
                priority="critical",
                detail=(
                    f"رُصد في {_zone_ar(zid)}: تدفق سريع بمقدار {mag:.2f} بكسل/إطار "
                    f"(الحد: {SMALL_FLOW_SURGE}) مع كثافة {d:.3f} شخص/م². "
                    f"اتجاه الحركة: {update.flow_direction:.0f}°. "
                    f"التحقق الفوري من خطر التدافع."
                ),
                public_message=(
                    f"🚨 رُصد تدفق سريع للحشود في {_zone_ar(zid)}. "
                    f"يرجى التحرك ببطء والالتزام بتوجيهات فريق السلامة."
                ),
            ))

        # Stagnation: near-zero flow in a dense zone signals a bottleneck
        # 0.30 p/sqm = ~120 people in 400 sqm stopped moving — serious risk
        if mag < 0.3 and d > SMALL_STAGNATION_MIN_DENSITY:
            actions.append(AgentAction(
                action="WARNING: Crowd stagnation detected — possible bottleneck forming",
                zone_id=zid,
                priority="high",
                detail=(
                    f"رُصد في {_zone_ar(zid)}: تدفق شبه منعدم ({mag:.2f} بكسل/إطار) "
                    f"مع كثافة {d:.3f} شخص/م² — توقف الحشد عن الحركة. "
                    f"التحقق من وجود عائق أو انسداد في المخارج."
                ),
                public_message=(
                    f"⚠️ يشهد {_zone_ar(zid)} توقفاً في حركة الحشود. "
                    f"يُرجى اتباع إرشادات المشرفين واختيار مسار بديل."
                ),
            ))

        # ── Cross-zone overload (requires ZoneStateCache) ─────────────────
        # 0.25 p/sqm = zone is clearly busy; if ALL neighbours are also >0.15
        # there is no safe overflow destination — venue-wide intervention needed.
        if d > SMALL_DENSITY_OVERLOAD and self._zone_cache is not None:
            adjacent = self._zone_cache.get_adjacent_zones(zid)
            if adjacent and all(a.density > SMALL_ADJACENT_PRESSURE for a in adjacent):
                neighbour_summary = ", ".join(
                    f"{a.zone_id}={a.density:.3f}" for a in adjacent
                )
                actions.append(AgentAction(
                    action="SYSTEM_OVERLOAD: Multiple zones under pressure — consider venue-wide intervention",
                    zone_id=zid,
                    priority="critical",
                    detail=(
                        f"{_zone_ar(zid)} (الكثافة: {d:.3f} شخص/م²) وجميع "
                        f"المناطق المجاورة ({len(adjacent)}) [{neighbour_summary}] "
                        f"تتجاوز {SMALL_ADJACENT_PRESSURE} شخص/م². "
                        f"لا توجد منطقة متاحة لتحويل الحشد — تصعيد فوري لعمليات الملعب."
                    ),
                    public_message=(
                        "🚨 تنبيه عام: تشهد مناطق متعددة في الملعب ضغطاً عالياً على الطاقة الاستيعابية. "
                        "يُرجى البقاء في أماكنكم واتباع توجيهات فريق التنظيم."
                    ),
                ))

        return actions

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
