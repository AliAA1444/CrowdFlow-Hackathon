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
from core.schemas import ZoneUpdate
from core.zone_state_cache import ZoneStateCache
from agents.base_agent import AgentAction, BaseAgent, run_agent

# ── Channels ─────────────────────────────────────────────────────────────────
CH_POS_TX       = "pos:transactions"
CH_ZONE_METRICS = "broadcast:zone_metrics"
CH_FAN_INTENT   = "fan:food_intent"
CH_ZONE_UPDATES = "zone_updates"

# ── Tunables (legacy POS / intent handlers) ──────────────────────────────────
HIGH_VELOCITY_THRESHOLD  = 40     # txns per minute triggers surge pricing
VELOCITY_WINDOW          = 60.0   # seconds to measure tx velocity
LOW_TRAFFIC_UTILIZATION  = 0.25   # zone utilization below this = low traffic
FLASH_DEAL_COOLDOWN      = 120.0  # seconds between flash deals for same zone
INTENT_THRESHOLD         = 15     # food intents needed to trigger promotion
INTENT_WINDOW            = 60.0   # seconds for intent accumulation

# ── Small-venue thresholds for evaluate() ────────────────────────────────────
# Threshold 0.12: density is low but people are moving — passing traffic, not dwellers.
# (previously 0.10; raised ~20% for imgsz=1280/conf=0.15 detection improvement)
SMALL_FLASH_MAX_DENSITY   = 0.12
# Threshold 1.8 px/frame: visible foot traffic above noise floor
# (previously 1.5; raised to stay above normal idle-crowd drift after CV fix)
SMALL_FLASH_MIN_FLOW      = 1.8

# Threshold 0.30: ~120 people in 400 sqm — zone is congested, don't attract more
# (previously 0.22; scaled up ~36% proportionally with detection improvement)
SMALL_PAUSE_DENSITY       = 0.30

# Threshold 25: gate is actively processing fans — good welcome-deal moment
# (previously 20; raised because higher detection captures more partial crossings)
SMALL_GATE_INFLUX         = 25
# Threshold 0.18: gate isn't yet congested — promotion won't cause a jam
# (previously 0.15; slight increase consistent with overall threshold recalibration)
SMALL_GATE_MAX_DENSITY    = 0.18

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


# Exterior zones that can act as overflow attractors
_EXTERIOR_OVERFLOW_ZONES = ["zone_activation_1", "zone_activation_2"]

# Zone IDs that have concession stalls (derived from config at import time)
_CONCESSION_ZONE_IDS: set[str] = {cfg["zone"] for cfg in CONCESSION_STALLS.values()}


class ConcessionAgent(BaseAgent):
    name = "concession_agent"
    subscribe_channels = [CH_POS_TX, CH_ZONE_METRICS, CH_FAN_INTENT, CH_ZONE_UPDATES]

    def __init__(self, zone_state_cache: ZoneStateCache | None = None) -> None:
        super().__init__()
        self._zone_cache = zone_state_cache
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
        elif channel == CH_ZONE_UPDATES:
            update = ZoneUpdate(**data)
            actions = await self.evaluate(update)
            await self.publish_actions(actions)

    # ── ZoneUpdate evaluation ────────────────────────────────────────────────

    async def evaluate(self, update: ZoneUpdate) -> list[AgentAction]:
        """Revenue-optimisation rules driven by live ZoneUpdate data.

        Thresholds calibrated for small-venue footage (zones 40–1200 sqm,
        typical counts 30–150 people, density range 0.05–0.40 p/sqm).

        Note: the old _CONCESSION_ZONE_IDS guard is removed here — evaluate()
        runs on all zones because our demo zones (gate_main, concourse_main,
        stands_east) are not listed in the legacy CONCESSION_STALLS config.
        The other methods (_handle_transaction etc.) still use CONCESSION_STALLS.
        """
        guard = await super().evaluate(update)
        if update.source == "stale_fallback":
            return guard

        actions: list[AgentAction] = []
        zid = update.zone_id
        d   = update.density
        mag = update.flow_magnitude

        # ── Flash sale: passing traffic, low dwell ────────────────────────
        # density < 0.10: zone is not crowded (people moving through)
        # flow > 1.5 px/frame: measurable foot traffic above noise floor
        if d < SMALL_FLASH_MAX_DENSITY and mag > SMALL_FLASH_MIN_FLOW:
            actions.append(AgentAction(
                action="FLASH_SALE",
                zone_id=zid,
                priority="low",
                detail=(
                    f"فرصة في {_zone_ar(zid)}: حركة مشاة مرتفعة "
                    f"(تدفق: {mag:.1f} بكسل/إطار) مع إقامة منخفضة "
                    f"(كثافة: {d:.3f} شخص/م²). "
                    f"إطلاق إشعار فوري عبر التطبيق للمطاعم القريبة."
                ),
                public_message=(
                    f"🎉 عرض حصري الآن! استمتع بخصم خاص على المشتريات "
                    f"بالقرب من {_zone_ar(zid)}. لا تفوت الفرصة!"
                ),
            ))

        # ── Pause promotions: zone is congested ───────────────────────────
        # density > 0.22: ~88 people in 400 sqm — don't attract more foot traffic
        elif d > SMALL_PAUSE_DENSITY:
            actions.append(AgentAction(
                action="PAUSE_PROMOTIONS",
                zone_id=zid,
                priority="medium",
                detail=(
                    f"ازدحام في {_zone_ar(zid)} (الكثافة: {d:.3f} شخص/م²). "
                    f"تعليق الإشعارات الترويجية لتجنب زيادة التكدس "
                    f"في منطقة مشغولة بالفعل."
                ),
            ))

        # ── Welcome deal: fans arriving at gate, zone still manageable ────
        # gate_in > 20: meaningful inflow occurring
        # density < 0.15: gate area not yet jammed — safe to push a promo
        if zid.startswith("gate_") and update.gate_in > SMALL_GATE_INFLUX \
                and d < SMALL_GATE_MAX_DENSITY:
            actions.append(AgentAction(
                action="WELCOME_DEAL",
                zone_id=zid,
                priority="low",
                detail=(
                    f"تدفق مشجعين في {_zone_ar(zid)}: {update.gate_in} دخولاً، "
                    f"الكثافة الحالية {d:.3f} شخص/م². "
                    f"إرسال عرض ترحيبي للقادمين الجدد عبر التطبيق."
                ),
                public_message=(
                    f"🎁 أهلاً وسهلاً بكم في CrowdFlow Arena! "
                    f"استمتعوا بعرض ترحيبي حصري عند {_zone_ar(zid)} — متاح الآن فقط!"
                ),
            ))

        return actions

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
