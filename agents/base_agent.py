from __future__ import annotations

"""CrowdFlow — Abstract base agent.

Every CrowdFlow agent inherits from BaseAgent, which handles Redis
connection, pub/sub subscription, message dispatch, and graceful
shutdown.

Not runnable standalone — subclass and implement ``process()``.
"""

import abc
import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path

import redis.asyncio as aioredis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import REDIS_URL

# Channel where agents submit decisions for the orchestrator
CH_ORCHESTRATOR = "orchestrator:decisions"


class BaseAgent(abc.ABC):
    """Base class for all CrowdFlow agents."""

    # Subclasses set these
    name: str = "base_agent"
    subscribe_channels: list[str] = []

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._running = True
        self.log = logging.getLogger(self.name)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self.redis.ping()
            self.log.info("Connected to Redis at %s", REDIS_URL)
        except Exception as exc:
            self.log.error("Redis connection failed: %s", exc)
            raise

    async def start(self) -> None:
        """Connect, subscribe, and enter the message loop."""
        await self.connect()
        self._pubsub = self.redis.pubsub()

        if not self.subscribe_channels:
            self.log.warning("No channels to subscribe to")
            return

        try:
            await self._pubsub.subscribe(*self.subscribe_channels)
            self.log.info(
                "Subscribed to %s", ", ".join(self.subscribe_channels),
            )
        except Exception as exc:
            self.log.error("Subscribe failed: %s", exc)
            raise

        try:
            async for msg in self._pubsub.listen():
                if not self._running:
                    break
                if msg["type"] != "message":
                    continue
                channel = msg["channel"]
                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    self.log.warning("Bad JSON on %s", channel)
                    continue
                try:
                    await self.process(channel, data)
                except Exception as exc:
                    self.log.error(
                        "Error processing %s: %s", channel, exc, exc_info=True,
                    )
        except asyncio.CancelledError:
            self.log.info("Message loop cancelled")
        except Exception as exc:
            self.log.error("Listener error: %s", exc)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        self._running = False
        self.log.info("Shutting down")
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe()
                await self._pubsub.aclose()
            except Exception:
                pass
        if self.redis:
            try:
                await self.redis.aclose()
            except Exception:
                pass

    # ── Abstract ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def process(self, channel: str, data: dict) -> None:
        """Handle an incoming message. Subclasses must implement this."""

    # ── Publishing helpers ───────────────────────────────────────────────────

    async def publish_decision(self, decision: dict) -> None:
        """Send a decision to the orchestrator channel.

        *decision* should include at minimum:
          agent, action, priority, zone_id, details
        """
        decision.setdefault("agent", self.name)
        decision.setdefault("timestamp", time.time())
        try:
            await self.redis.publish(CH_ORCHESTRATOR, json.dumps(decision))
            self.log.info(
                "Decision → %s (pri %s) for %s",
                decision.get("action"),
                decision.get("priority"),
                decision.get("zone_id", "global"),
            )
        except Exception as exc:
            self.log.error("publish_decision failed: %s", exc)

    async def publish_to(self, channel: str, data: dict) -> None:
        """Publish arbitrary JSON to any channel."""
        try:
            await self.redis.publish(channel, json.dumps(data))
        except Exception as exc:
            self.log.error("publish_to %s failed: %s", channel, exc)


# ── Standalone runner helper ─────────────────────────────────────────────────

def run_agent(agent_cls: type[BaseAgent]) -> None:
    """Convenience entry-point for running an agent from the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = agent_cls()

    loop = asyncio.new_event_loop()

    def _stop(*_: object) -> None:
        agent._running = False
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    try:
        loop.run_until_complete(agent.start())
    finally:
        loop.close()
