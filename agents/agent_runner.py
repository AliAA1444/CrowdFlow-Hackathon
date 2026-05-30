from __future__ import annotations

"""CrowdFlow — AgentRunner.

Centralised orchestrator that:
  1. Subscribes to the Redis ``zone_updates`` channel.
  2. Parses each message as a ZoneUpdate and updates ZoneStateCache.
  3. Runs SafetyAgent, CrowdFlowAgent, and ConcessionAgent evaluate() in order.
  4. Uses CooldownManager to decide whether each proposed action should be
     published.  An action fires exactly once when its condition first becomes
     true, is suppressed while the condition persists (ACTIVE phase), and is
     allowed to re-trigger only after the condition clears AND a per-action-type
     cooldown period has elapsed.
  5. Clears previously-active actions whose conditions are no longer met so that
     cooldown timers start immediately when the scene recovers.
  6. Publishes qualifying AgentAction results to the ``agent_actions`` channel.
  7. Wraps every message iteration in try/except — a single bad message never
     kills the runner.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis

from core.schemas import ZoneUpdate
from core.zone_state_cache import ZoneStateCache
from agents.safety_agent import SafetyAgent
from agents.crowd_flow_agent import CrowdFlowAgent
from agents.concession_agent import ConcessionAgent
from agents.cooldown_manager import CooldownManager

# FIX: channel name matches redis_publisher.CHANNEL_ZONE_UPDATES = "zone_updates"
CHANNEL_ZONE_UPDATES  = "zone_updates"
CHANNEL_AGENT_ACTIONS = "agent_actions"

logger = logging.getLogger("crowdflow.agent_runner")


class AgentRunner:
    """Orchestrates all three agents against a shared Redis stream."""

    def __init__(self, redis: aioredis.Redis, cache: ZoneStateCache) -> None:
        self._redis = redis
        self._cache = cache

        # Pass ZoneStateCache so cross-zone lookups work in every agent.
        self._agents: list[tuple[str, SafetyAgent | CrowdFlowAgent | ConcessionAgent]] = [
            ("safety_agent",    SafetyAgent(cache)),
            ("crowd_flow_agent", CrowdFlowAgent(cache)),
            ("concession_agent", ConcessionAgent(cache)),
        ]
        for _, agent in self._agents:
            agent.redis = self._redis

        # State-machine cooldown manager — replaces the old flat _last_action dict.
        self.cooldown_manager = CooldownManager()

        # Timestamp of the last periodic stats log (avoid repeated logs on same second).
        self._last_stats_log: float = 0.0

        logger.info(
            "AgentRunner started — Safety, CrowdFlow, Concession agents active. "
            "Rate limit: 3s"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Subscribe to zone_updates, evaluate all agents, publish decisions.

        Designed to run as a long-lived asyncio task; cancel to stop.
        """
        sub = self._redis.pubsub()
        try:
            await sub.subscribe(CHANNEL_ZONE_UPDATES)
            logger.info(
                "AgentRunner subscribed to '%s' — running %d agents",
                CHANNEL_ZONE_UPDATES,
                len(self._agents),
            )

            async for msg in sub.listen():
                if msg["type"] != "message":
                    continue

                try:
                    data   = json.loads(msg["data"])
                    update = ZoneUpdate(**data)
                    self._cache.update(update)
                    await self._evaluate_all(update)
                except Exception:
                    logger.exception(
                        "AgentRunner: unhandled error processing message — continuing"
                    )

        except asyncio.CancelledError:
            logger.info("AgentRunner stopped")
        except Exception as exc:
            logger.error("AgentRunner listener error: %s", exc)
        finally:
            try:
                await sub.unsubscribe()
                await sub.aclose()
            except Exception:
                pass

    # ── Internal ───────────────────────────���───────────────────���──────────────

    async def _evaluate_all(self, update: ZoneUpdate) -> None:
        """Run every agent, apply cooldown logic, publish qualifying actions."""
        now = time.time()

        # ── Step 1: collect all proposed actions from every agent ─────────
        # Stored as (agent_name, AgentAction) so we can tag the payload.
        proposed: list[tuple[str, object]] = []
        for agent_name, agent in self._agents:
            try:
                actions = await agent.evaluate(update)
                if actions:
                    for action in actions:
                        proposed.append((agent_name, action))
            except Exception:
                logger.exception(
                    "Agent %s raised during evaluate() for zone %s",
                    agent_name,
                    update.zone_id,
                )

        # ── Step 2: clear actions whose conditions are no longer met ──────
        # Any action that was ACTIVE for this zone but was NOT proposed this
        # cycle means the triggering condition has resolved — start cooldown.
        proposed_keys = {
            action.action
            for _, action in proposed
            if action.zone_id == update.zone_id
        }
        for state_key, state in list(self.cooldown_manager._states.items()):
            if state["zone_id"] == update.zone_id and state["status"] == "active":
                if state["action"] not in proposed_keys:
                    self.cooldown_manager.mark_cleared(state["action"], state["zone_id"])

        # ── Step 3: publish qualifying actions via cooldown gate ──────────
        for agent_name, action in proposed:
            if not self.cooldown_manager.should_fire(action.action, action.zone_id):
                continue

            payload: dict = {
                "agent":     agent_name,
                "action":    action.action,
                "zone_id":   action.zone_id,
                "priority":  action.priority,
                "detail":    action.detail,
                "timestamp": now,
            }
            if action.public_message is not None:
                payload["public_message"] = action.public_message
            try:
                await self._redis.publish(
                    CHANNEL_AGENT_ACTIONS, json.dumps(payload)
                )
                logger.info(
                    "[%s] %s → zone=%s pri=%s",
                    agent_name,
                    action.action,
                    action.zone_id,
                    action.priority,
                )
            except Exception as exc:
                logger.error(
                    "AgentRunner: publish to %s failed: %s",
                    CHANNEL_AGENT_ACTIONS,
                    exc,
                )

        # ── Step 4: periodic stats log ────────────────────────────────────
        if now - self._last_stats_log >= 60.0:
            stats = self.cooldown_manager.get_stats()
            logger.info(
                "CooldownManager: active=%d tracked=%d fires=%d suppressed=%d",
                stats["active"],
                stats["total_tracked"],
                stats.get("total_fires", 0),
                stats.get("total_suppressed", 0),
            )
            self._last_stats_log = now
