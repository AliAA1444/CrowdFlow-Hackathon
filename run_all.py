from __future__ import annotations

"""CrowdFlow — Master launcher.

Starts camera inference, fallback simulation, all agents, and the
FastAPI backend with a single command:  python run_all.py

Camera inference (YOLOv8-nano) runs for every zone that has a video
file present in data/videos/.  The fallback simulator covers any zone
whose video file is missing, so the demo works even with zero videos.
"""

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CAMERA_FEEDS, ZONES
from simulation.pos_simulator import POSSimulator
from inference.camera_streamer import CameraInferenceEngine
from inference.fallback_simulator import FallbackSimulator
from agents.crowd_flow_agent import CrowdFlowAgent
from agents.safety_agent import SafetyAgent
from agents.concession_agent import ConcessionAgent
from agents.orchestrator import Orchestrator

UVICORN_PORT = 8000


# ── Banner (built at runtime so it can show live file-presence info) ──────────

def _build_banner() -> str:
    total_feeds = len(CAMERA_FEEDS)
    found, missing, inference_zones, fallback_zones = [], [], [], []

    # Build a set of zone_ids covered by live cameras
    live_zone_ids: set[str] = set()
    for cam_id, cfg in CAMERA_FEEDS.items():
        vf = cfg["video_file"]
        if os.path.exists(vf):
            found.append(cam_id)
            live_zone_ids.add(cfg["zone_id"])
        else:
            missing.append(cam_id)

    for zid in ZONES:
        if zid in live_zone_ids:
            inference_zones.append(zid)
        else:
            fallback_zones.append(zid)

    found_n   = len(found)
    missing_n = len(missing)

    inf_list = "  ".join(inference_zones) if inference_zones else "(none)"
    fb_list  = "  ".join(fallback_zones)  if fallback_zones  else "(none)"

    # Truncate long lists so the banner stays readable
    def _fmt(lst: list[str], limit: int = 4) -> str:
        if len(lst) <= limit:
            return ", ".join(lst)
        return ", ".join(lst[:limit]) + f" (+{len(lst)-limit} more)"

    return f"""
================================================================================
   ██████╗██████╗  ██████╗ ██╗    ██╗██████╗ ███████╗██╗      ██████╗ ██╗    ██╗
  ██╔════╝██╔══██╗██╔═══██╗██║    ██║██╔══██╗██╔════╝██║     ██╔═══██╗██║    ██║
  ██║     ██████╔╝██║   ██║██║ █╗ ██║██║  ██║█████╗  ██║     ██║   ██║██║ █╗ ██║
  ██║     ██╔══██╗██║   ██║██║███╗██║██║  ██║██╔══╝  ██║     ██║   ██║██║███╗██║
  ╚██████╗██║  ██║╚██████╔╝╚███╔███╔╝██████╔╝██║     ███████╗╚██████╔╝╚███╔███╔╝
   ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚══╝╚══╝ ╚═════╝ ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝
  Multi-Agent Stadium Intelligence System  ·  Riyadh 2026
================================================================================

  CAMERA FEEDS : {total_feeds} configured  ·  {found_n} video files found  ·  {missing_n} missing
  INFERENCE    : {_fmt(inference_zones)}
  FALLBACK SIM : {_fmt(fallback_zones)}

  SERVICES:
    [CAM] Camera Inference ............ YOLOv8-nano on {found_n}/{total_feeds} feeds
    [SIM] Fallback Simulator .......... covering {len(fallback_zones)} zone(s) without video
    [SIM] POS Simulator ............... publishing to pos:transactions
    [AGT] Safety Agent ................ priority 0  (crush-risk + medical)
    [AGT] Crowd Flow Agent ............ priority 2  (rerouting + overflow)
    [AGT] Concession Agent ............ priority 3  (revenue + exterior deals)
    [AGT] Orchestrator ................ conflict resolution engine
    [API] FastAPI Backend ............. http://localhost:{UVICORN_PORT}

  FRONTENDS:
    Dashboard ......................... http://localhost:{UVICORN_PORT}/dashboard
    Fan Web App ....................... http://localhost:{UVICORN_PORT}/fan

  DEMO CONTROLS (curl):
    Halftime surge:
      curl -X POST http://localhost:{UVICORN_PORT}/api/demo/trigger-halftime
    Emergency:
      curl -X POST http://localhost:{UVICORN_PORT}/api/demo/trigger-emergency
    Medical emergency:
      curl -X POST http://localhost:{UVICORN_PORT}/api/demo/trigger-medical \\
           -H "Content-Type: application/json" \\
           -d '{{"zone_id":"zone_north_concourse","reason":"Fan collapsed"}}'
    Reset:
      curl -X POST http://localhost:{UVICORN_PORT}/api/demo/reset

  Press Ctrl+C to stop all services.
================================================================================
"""


# ── Task wrappers ─────────────────────────────────────────────────────────────

async def run_service(coro_or_instance, name: str, *, method: str = "run") -> None:
    """Run any service that exposes .run() or .start(); catch errors gracefully."""
    try:
        fn = getattr(coro_or_instance, method)
        await fn()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        print(f"[ERROR] {name} crashed: {exc}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(_build_banner())

    # ── FastAPI via uvicorn subprocess ────────────────────────────────────────
    uvicorn_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", str(UVICORN_PORT),
            "--log-level", "warning",
        ],
        cwd=str(ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # ── Service instances ─────────────────────────────────────────────────────
    camera_engine = CameraInferenceEngine()
    fallback_sim  = FallbackSimulator()
    pos_sim       = POSSimulator()

    safety        = SafetyAgent()
    crowd_flow    = CrowdFlowAgent()
    concession    = ConcessionAgent()
    orchestrator  = Orchestrator()

    # ── Launch all async tasks ────────────────────────────────────────────────
    tasks = [
        # Inference layer
        asyncio.create_task(run_service(camera_engine, "CameraInferenceEngine")),
        asyncio.create_task(run_service(fallback_sim,  "FallbackSimulator")),
        asyncio.create_task(run_service(pos_sim,       "POSSimulator")),
        # Agent layer (orchestrator first so it's ready to receive decisions)
        asyncio.create_task(run_service(orchestrator,  "Orchestrator",  method="start")),
        asyncio.create_task(run_service(safety,        "SafetyAgent",   method="start")),
        asyncio.create_task(run_service(crowd_flow,    "CrowdFlowAgent",method="start")),
        asyncio.create_task(run_service(concession,    "ConcessionAgent",method="start")),
    ]

    # ── Graceful Ctrl+C / SIGTERM shutdown ────────────────────────────────────
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        print("\n[SHUTDOWN] Stopping all services...", flush=True)
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        if uvicorn_proc.poll() is None:
            uvicorn_proc.terminate()
            try:
                uvicorn_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                uvicorn_proc.kill()
        print("[SHUTDOWN] All services stopped.")


if __name__ == "__main__":
    asyncio.run(main())
