#!/usr/bin/env python3
"""CrowdFlow — Demo Reset Utility.

Flushes all zone:* keys from Redis so the next demo presentation
starts with a clean slate.

Usage:
    python tools/demo_reset.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import redis.asyncio as aioredis

from config import REDIS_URL


async def reset() -> None:
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        # Collect all zone:* keys
        keys: list[str] = []
        async for key in client.scan_iter(match="zone:*"):
            keys.append(key)

        if keys:
            deleted = await client.delete(*keys)
            print(f"[RESET] Deleted {deleted} zone key(s): {', '.join(sorted(keys))}")
        else:
            print("[RESET] No zone:* keys found — Redis is already clean.")

        # Also clear the demo mode flag if set
        demo_deleted = await client.delete("demo:mode")
        if demo_deleted:
            print("[RESET] Cleared demo:mode key.")

        print("\nRedis state cleared. Ready for fresh demo.")
    except Exception as exc:
        print(f"[ERROR] Could not connect to Redis: {exc}", file=sys.stderr)
        print("Make sure Redis is running: redis-server --daemonize yes", file=sys.stderr)
        sys.exit(1)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(reset())
