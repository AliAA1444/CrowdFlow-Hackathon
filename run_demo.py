#!/usr/bin/env python3
"""CrowdFlow — Master Demo Launch Script.

Starts the complete demo with a single command:
    python run_demo.py
    python run_demo.py --loop
    python run_demo.py --loop --no-calibration

Components started (in order):
  1. Redis liveness check
  2. Video file verification
  3. VisionPipeline  (4 video feeds → Redis)
  4. StalenessMonitor (background watchdog)
  5. Safety / CrowdFlow / Concession agents
  6. FastAPI + Uvicorn  (http://localhost:8000)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Project root on sys.path so all local imports resolve ────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import redis.asyncio as aioredis
import uvicorn

from backend.main import app
from core.staleness import StalenessMonitor
from vision.pipeline import VisionPipeline
from agents.safety_agent import SafetyAgent
from agents.crowd_flow_agent import CrowdFlowAgent
from agents.concession_agent import ConcessionAgent
from config import REDIS_URL

ZONES_JSON   = ROOT / "config" / "zones.json"
LOOP_REPEAT  = 5   # number of times each video is concatenated when --loop is set


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrowdFlow Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Pre-process each video into a %(repeat)dx concatenated file via ffmpeg "
            "for seamless continuous looping during extended demos." % {"repeat": LOOP_REPEAT}
        ),
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        dest="no_calibration",
        help=(
            "Freeze cross-calibration at the manual baseline factors from zones.json. "
            "Useful as a safety fallback when live ground-truth is unreliable."
        ),
    )
    return parser.parse_args()


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_redis() -> None:
    """Verify Redis is reachable via a raw socket probe (no async needed here)."""
    import socket
    try:
        sock = socket.create_connection(("localhost", 6379), timeout=2)
        sock.close()
    except OSError:
        print(
            "\n[ERROR] Redis is not running. Start it with:\n"
            "  redis-server --daemonize yes\n"
        )
        sys.exit(1)


def _load_zone_config() -> dict:
    """Load zones.json and abort if any video file is missing."""
    with open(ZONES_JSON) as fh:
        zone_config = json.load(fh)

    missing: list[str] = []
    for zone_id, cfg in zone_config["zones"].items():
        video_path = ROOT / cfg.get("video_source", "")
        if not video_path.exists():
            missing.append(f"  [{zone_id}]  {video_path}")

    if missing:
        print("\n[ERROR] Missing video files — cannot start demo:")
        for m in missing:
            print(m)
        print("\nPlace the video files at the paths above and retry.\n")
        sys.exit(1)

    return zone_config


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner(calibration_enabled: bool) -> None:
    cal_status = "ENABLED " if calibration_enabled else "DISABLED"
    print(f"""
╔══════════════════════════════════════════════════╗
║           CrowdFlow MVP — Demo Mode              ║
║                                                  ║
║  Zones:  4 (2 gates + 2 general)                ║
║  Feeds:  4 real video files                      ║
║  Agents: Safety, CrowdFlow, Concession          ║
║  Cross-Calibration: {cal_status}                 ║
╚══════════════════════════════════════════════════╝

  Dashboard  →  http://localhost:8000/dashboard
  Fan app    →  http://localhost:8000/fan
  API docs   →  http://localhost:8000/docs

  Press Ctrl+C to stop.
""")


# ── Looped video preparation ──────────────────────────────────────────────────

def _prepare_looped_feeds(
    zone_config: dict,
    tmpdir: str,
    repeat: int = LOOP_REPEAT,
) -> dict[str, str]:
    """Concatenate each video *repeat* times into *tmpdir* using ffmpeg.

    Returns a feed_config dict mapping zone_id → absolute video path.
    Falls back to the original path for any zone where ffmpeg fails.
    """
    if not shutil.which("ffmpeg"):
        print("[WARN] ffmpeg not found — --loop has no effect (feeds use original files)\n")
        return {
            zid: str(ROOT / cfg["video_source"])
            for zid, cfg in zone_config["zones"].items()
        }

    feed_config: dict[str, str] = {}
    for zone_id, cfg in zone_config["zones"].items():
        src  = str(ROOT / cfg["video_source"])
        out  = os.path.join(tmpdir, f"{zone_id}_loop{repeat}x.mp4")
        # ffmpeg concat demuxer list file
        clist = os.path.join(tmpdir, f"{zone_id}_concat.txt")
        with open(clist, "w") as f:
            for _ in range(repeat):
                f.write(f"file '{src}'\n")

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", clist,
                "-c", "copy",
                out,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  [LOOP] {zone_id}: {repeat}x copy → {out}")
            feed_config[zone_id] = out
        else:
            print(f"  [WARN] ffmpeg failed for {zone_id}, using original video")
            feed_config[zone_id] = src

    return feed_config


# ── Service runners ───────────────────────────────────────────────────────────

async def _run_agent(agent_cls, name: str) -> None:
    """Instantiate and start an agent; absorb CancelledError on shutdown."""
    try:
        agent = agent_cls()
        await agent.start()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        print(f"[ERROR] {name} crashed: {exc}", file=sys.stderr)


# ── Main entry point ──────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    # 1. Redis must be up before anything else.
    print("[STARTUP] Checking Redis...", flush=True)
    _check_redis()
    print("[STARTUP] Redis OK")

    # 2. Verify zone config and all four video files.
    print("[STARTUP] Verifying zone config and video files...", flush=True)
    zone_config = _load_zone_config()
    print(f"[STARTUP] {len(zone_config['zones'])} zones verified")

    # 3. Print startup banner.
    _print_banner(calibration_enabled=not args.no_calibration)

    # 4. Prepare feed_config (possibly looped via ffmpeg).
    tmpdir: str | None = None
    if args.loop:
        tmpdir = tempfile.mkdtemp(prefix="crowdflow_loop_")
        print(f"[LOOP] Pre-processing {LOOP_REPEAT}x looped videos in {tmpdir} ...")
        feed_config = _prepare_looped_feeds(zone_config, tmpdir)
        print()
    else:
        feed_config = {
            zid: str(ROOT / cfg["video_source"])
            for zid, cfg in zone_config["zones"].items()
        }

    # 5. Build VisionPipeline (loads YOLO model, sets up per-zone components).
    print("[STARTUP] Initialising VisionPipeline...", flush=True)
    pipeline = VisionPipeline(
        yolo_model_path=str(ROOT / "yolov8n.pt"),
        zone_config=zone_config,
        redis_url=REDIS_URL,
    )

    # 6. Apply --no-calibration: freeze each CrossCalibrator at its initial factor.
    # Monkey-patching update() means learning_rate is irrelevant and _confidence
    # stays at 0, so the source label remains "cv_dense" (no "cal@" suffix).
    if args.no_calibration:
        print("[CONFIG] Cross-calibration DISABLED — pinning to manual baseline factors")
        for zone_id, cal in pipeline.cross_calibrators.items():
            frozen = cal._current_factor
            # Lambda default-captures frozen to avoid late-binding closure issues.
            cal.update = lambda yc, fp, f=frozen: f  # type: ignore[method-assign]
    else:
        print("[CONFIG] Cross-calibration ENABLED")

    # 7. Shared Redis client for the StalenessMonitor.
    redis_client = aioredis.from_url(REDIS_URL)
    staleness_monitor = StalenessMonitor(redis_client)

    # 8. Uvicorn server (programmatic API — no subprocess).
    uv_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server    = uvicorn.Server(uv_config)

    # ── Gather all concurrent tasks ───────────────────────────────────────────
    all_tasks: list[asyncio.Task] = []

    async def _run_staleness() -> None:
        try:
            await staleness_monitor.run()
        except asyncio.CancelledError:
            pass

    async def _run_pipeline() -> None:
        try:
            await pipeline.run_all_feeds(feed_config)
        except asyncio.CancelledError:
            pass

    async def _run_server() -> None:
        try:
            await server.serve()
        except asyncio.CancelledError:
            pass

    loop = asyncio.get_running_loop()

    def _shutdown(*_) -> None:
        print("\n[SHUTDOWN] Shutting down CrowdFlow...", flush=True)
        server.should_exit = True   # ask uvicorn to stop gracefully
        for task in all_tasks:
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    all_tasks.extend([
        asyncio.create_task(_run_staleness(),                          name="staleness_monitor"),
        asyncio.create_task(_run_pipeline(),                           name="vision_pipeline"),
        asyncio.create_task(_run_agent(SafetyAgent,     "SafetyAgent"),    ),
        asyncio.create_task(_run_agent(CrowdFlowAgent,  "CrowdFlowAgent"), ),
        asyncio.create_task(_run_agent(ConcessionAgent, "ConcessionAgent"),),
        asyncio.create_task(_run_server(),                             name="uvicorn_server"),
    ])

    try:
        await asyncio.gather(*all_tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await redis_client.aclose()
        except Exception:
            pass
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"[CLEANUP] Removed temp loop dir: {tmpdir}")
        print("[SHUTDOWN] All services stopped.")


if __name__ == "__main__":
    asyncio.run(main())
