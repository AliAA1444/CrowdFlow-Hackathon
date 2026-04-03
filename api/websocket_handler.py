from __future__ import annotations

"""CrowdFlow — Dashboard WebSocket handler (Prompt 5).

A single shared Redis pub/sub listener that fans out to every connected
dashboard WebSocket client.  Keeping one Redis connection for N clients
is far more efficient than one connection per client.

Channels subscribed
-------------------
zone_updates          →  {"event": "zone_update",  "data": <ZoneUpdate dict>}
zone_stale            →  {"event": "zone_stale",   "zone_id": ..., "stale_since": ...}
agent_actions         →  {"event": "agent_action", "data": <AgentAction dict>}
dashboard:actions     →  {"channel": "dashboard:actions",     "data": ...}  (legacy)
pos:match_clock       →  {"channel": "pos:match_clock",       "data": ...}  (legacy)
broadcast:zone_metrics→  {"channel": "broadcast:zone_metrics","data": ...}  (legacy)

On new client connect the handler sends the most-recent ZoneUpdate for
every zone from the in-memory cache so the dashboard is never blank.
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("crowdflow.ws_handler")

# ── Channels ─────────────────────────────────────────────────────────────────
_CH_ZONE_UPDATES  = "zone_updates"
_CH_ZONE_STALE    = "zone_stale"
_CH_AGENT_ACTIONS = "agent_actions"

# Legacy channels — forwarded with original {channel, data} envelope
_CH_ZONE_METRICS  = "broadcast:zone_metrics"
_CH_MATCH_CLOCK   = "pos:match_clock"
_CH_DASH_ACTIONS  = "dashboard:actions"


class DashboardWSHandler:
    """Broadcast hub for all connected dashboard WebSocket clients."""

    def __init__(self) -> None:
        # zone_id → last ZoneUpdate dict (catch-up state for new connections)
        self._zone_cache: dict[str, dict] = {}
        # All currently live WebSocket connections
        self._clients: set[WebSocket] = set()

    # ── Client lifecycle ─────────────────────────────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        """Accept *ws*, push cached zone state, then hold until disconnect."""
        await ws.accept()
        self._clients.add(ws)
        logger.info("Dashboard WS client connected (%d total)", len(self._clients))

        # Send each zone's last-known ZoneUpdate so the dashboard isn't blank
        for zone_data in list(self._zone_cache.values()):
            try:
                await ws.send_json({"event": "zone_update", "data": zone_data})
            except Exception:
                self._clients.discard(ws)
                return

        # Hold the connection open; incoming dashboard frames are ignored
        try:
            while True:
                await ws.receive_text()
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            self._clients.discard(ws)
            logger.info("Dashboard WS client disconnected (%d total)", len(self._clients))

    # ── Redis background listener ────────────────────────────────────────────

    async def run_redis_listener(self, redis: aioredis.Redis) -> None:
        """Subscribe to Redis and broadcast to all connected clients.

        Call once as a long-lived ``asyncio`` background task at app startup.
        """
        sub = redis.pubsub()
        try:
            await sub.subscribe(
                _CH_ZONE_UPDATES,
                _CH_ZONE_STALE,
                _CH_AGENT_ACTIONS,
                _CH_ZONE_METRICS,
                _CH_MATCH_CLOCK,
                _CH_DASH_ACTIONS,
            )
            logger.info("WS handler subscribed to Redis channels")

            async for msg in sub.listen():
                if msg["type"] != "message":
                    continue
                channel = msg["channel"]
                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Bad JSON on channel %s", channel)
                    continue

                if channel == _CH_ZONE_UPDATES:
                    # Cache and forward the full ZoneUpdate AS-IS
                    zone_id = data.get("zone_id")
                    if zone_id:
                        self._zone_cache[zone_id] = data
                    await self._broadcast({"event": "zone_update", "data": data})

                elif channel == _CH_ZONE_STALE:
                    await self._broadcast({
                        "event":       "zone_stale",
                        "zone_id":     data.get("zone_id"),
                        "stale_since": data.get("stale_since"),
                    })

                elif channel == _CH_AGENT_ACTIONS:
                    await self._broadcast({"event": "agent_action", "data": data})

                else:
                    # Legacy channels: preserve original {channel, data} envelope
                    await self._broadcast({"channel": channel, "data": data})

        except asyncio.CancelledError:
            logger.info("WS Redis listener cancelled")
        except Exception as exc:
            logger.error("WS Redis listener error: %s", exc)
        finally:
            try:
                await sub.unsubscribe()
                await sub.aclose()
            except Exception:
                pass

    # ── Broadcast ────────────────────────────────────────────────────────────

    async def _broadcast(self, payload: dict) -> None:
        """Send *payload* to every client; silently drop any that have errored."""
        dead: list[WebSocket] = []
        for client in list(self._clients):
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)


# Module-level singleton — imported by backend/main.py
handler = DashboardWSHandler()
