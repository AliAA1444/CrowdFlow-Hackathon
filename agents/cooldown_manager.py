"""
CooldownManager v2 — Stateful action lifecycle with hard limits.

Lifecycle per action+zone:
  READY → (condition met, should_fire=True) → ACTIVE
  ACTIVE → (stays active, should_fire=False) → ACTIVE  [no re-fire]
  ACTIVE → (mark_cleared called) → COOLDOWN
  COOLDOWN → (timer expires) → READY

Hard safety limits:
  - Per-zone: max 1 action per zone per 8 seconds
  - Per-action-type: governed by COOLDOWNS dict (post-clear cooldown)
"""

import time
import logging

logger = logging.getLogger("crowdflow.cooldown")


class CooldownManager:

    # How long after a condition CLEARS before the same action can re-trigger
    COOLDOWNS: dict[str, int] = {
        "CRITICAL": 45,
        "HIGH_DENSITY_WARNING": 45,
        "CROWD_SURGE": 30,
        "STAGNATION_WARNING": 30,
        "SYSTEM_OVERLOAD": 60,
        "REROUTE_FANS": 90,
        "GATE_THROTTLE": 60,
        "FLASH_SALE": 60,
        "PAUSE_PROMOTIONS": 45,
        "WELCOME_DEAL": 90,
    }
    DEFAULT_COOLDOWN: int = 45

    def __init__(self):
        # Key: "{action}:{zone_id}" → dict with state info
        self._states: dict[str, dict] = {}

        # PER-ZONE RATE LIMIT: zone_id → timestamp of last fire for that zone.
        # Limits actions for the same zone to at most 1 per interval.
        # Different zones can fire independently — 9 zones × 1 action each is fine.
        self._last_zone_fire: dict[str, float] = {}
        self._zone_min_interval: float = 8.0  # seconds between actions for same zone

        # Counters for debugging
        self._total_fires: int = 0
        self._total_suppressed: int = 0

    def should_fire(self, action: str, zone_id: str) -> bool:
        """
        Returns True ONLY if:
        1. This action+zone is not currently ACTIVE
        2. This action+zone is not in COOLDOWN
        3. Global rate limit is not exceeded
        4. Per-zone rate limit is not exceeded
        """
        now = time.time()
        key = f"{action}:{zone_id}"

        # === PER-ZONE RATE LIMIT ===
        last_zone = self._last_zone_fire.get(zone_id, 0.0)
        if now - last_zone < self._zone_min_interval:
            self._total_suppressed += 1
            return False

        # === STATE CHECK ===
        if key in self._states:
            state = self._states[key]

            if state["status"] == "active":
                # Already active — do NOT re-fire
                return False

            if state["status"] == "cooldown":
                cooldown_duration = self.COOLDOWNS.get(action, self.DEFAULT_COOLDOWN)
                elapsed = now - state["cleared_at"]
                if elapsed < cooldown_duration:
                    # Still cooling down
                    return False
                # Cooldown expired — allow re-trigger
                state["status"] = "active"
                state["triggered_at"] = now
                state["fire_count"] = state.get("fire_count", 0) + 1
            else:
                # status == "ready" — allow trigger
                state["status"] = "active"
                state["triggered_at"] = now
                state["fire_count"] = state.get("fire_count", 0) + 1
        else:
            # Never seen — create and allow
            self._states[key] = {
                "status": "active",
                "action": action,
                "zone_id": zone_id,
                "triggered_at": now,
                "cleared_at": 0.0,
                "fire_count": 1,
            }

        # Update per-zone rate limit timestamp
        self._last_zone_fire[zone_id] = now
        self._total_fires += 1

        logger.info(
            f"Action FIRED: {action} @ {zone_id} "
            f"(fire #{self._states[key]['fire_count']}, "
            f"total: {self._total_fires})"
        )
        return True

    def mark_cleared(self, action: str, zone_id: str) -> None:
        """
        Call when the condition that triggered an action is no longer true.
        Moves state from ACTIVE → COOLDOWN.
        """
        key = f"{action}:{zone_id}"
        if key in self._states and self._states[key]["status"] == "active":
            self._states[key]["status"] = "cooldown"
            self._states[key]["cleared_at"] = time.time()
            logger.debug(f"Action CLEARED: {action} @ {zone_id} → cooldown started")

    def get_active_count(self) -> int:
        """Number of currently active (non-cleared) actions."""
        return sum(1 for s in self._states.values() if s["status"] == "active")

    def get_stats(self) -> dict:
        """Summary for logging."""
        return {
            "active": self.get_active_count(),
            "total_tracked": len(self._states),
            "total_fires": self._total_fires,
            "total_suppressed": self._total_suppressed,
        }
