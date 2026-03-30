from __future__ import annotations

"""CrowdFlow — Camera inference engine.

Processes video files with YOLOv8-nano to count people per zone, then
publishes zone metrics to the same Redis channel the old crowd simulator
used (broadcast:zone_metrics).  All existing agents receive the same
payload structure and work without modification.

Run standalone:  python inference/camera_streamer.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import redis.asyncio as aioredis
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CAMERA_FEEDS,
    CH_CAMERA_INFERENCE,
    CONGESTION_STATES,
    REDIS_URL,
    ZONES,
    classify_congestion,
    get_state_config,
)

# Matches the channel name defined in crowd_simulator.py and all agent files.
# Redeclared here so this module stays self-contained.
CH_ZONE_METRICS = "broadcast:zone_metrics"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CAMERA] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("camera_streamer")


# ── Shared match clock ────────────────────────────────────────────────────────

class MatchClock:
    """Wall-clock that maps to simulated match minutes at 10× speed."""

    SPEED_MULTIPLIER = 10.0
    MATCH_DURATION   = 90.0   # minutes

    def __init__(self) -> None:
        self._start = time.monotonic()

    @property
    def match_minute(self) -> float:
        elapsed_real_s = time.monotonic() - self._start
        sim_minutes    = (elapsed_real_s * self.SPEED_MULTIPLIER) / 60.0
        return round(min(sim_minutes, self.MATCH_DURATION), 1)


# ── Inference engine ─────────────────────────────────────────────────────────

class CameraInferenceEngine:
    """Runs one async task per camera, feeds frames through YOLO, publishes metrics."""

    def __init__(self) -> None:
        log.info("Loading YOLOv8n model (auto-downloads on first run)…")
        self.model = YOLO("yolov8n.pt")
        log.info("YOLOv8n ready.")
        self.redis: aioredis.Redis | None = None
        self.clock = MatchClock()

    # ── Redis ────────────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await self.redis.ping()
        log.info("Connected to Redis at %s", REDIS_URL)

    async def _publish(self, channel: str, payload: dict) -> None:
        try:
            await self.redis.publish(channel, json.dumps(payload))
        except Exception as exc:
            log.warning("Publish to %s failed: %s", channel, exc)

    async def _store(self, key: str, payload: dict, ttl: int = 30) -> None:
        try:
            await self.redis.set(key, json.dumps(payload), ex=ttl)
        except Exception as exc:
            log.warning("Redis SET %s failed: %s", key, exc)

    # ── Per-camera loop ──────────────────────────────────────────────────────

    async def process_camera(self, camera_id: str, cfg: dict) -> None:
        """Read frames from a video file and publish zone metrics indefinitely."""
        zone_id    = cfg["zone_id"]
        video_file = cfg["video_file"]
        fps        = max(1, cfg.get("fps", 2))

        # ── Guard: skip if video missing ─────────────────────────────────────
        if not os.path.exists(video_file):
            log.warning(
                "Camera %s: %s not found — skipping (fallback_simulator will cover %s)",
                camera_id, video_file, zone_id,
            )
            return

        zone_cfg = ZONES.get(zone_id)
        if zone_cfg is None:
            log.warning("Camera %s: zone_id '%s' not in ZONES — skipping", camera_id, zone_id)
            return

        area_m2    = float(zone_cfg.get("area_m2", 1.0))
        capacity   = int(zone_cfg.get("capacity", 1))
        zone_label = zone_cfg.get("label", zone_id)

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            log.warning("Camera %s: cv2 could not open %s", camera_id, video_file)
            return

        log.info("Camera %s → zone '%s' | file: %s | fps: %d", camera_id, zone_id, video_file, fps)
        frame_interval = 1.0 / fps
        last_valid_count: int = 5   # hold last non-zero detection count; seed with 5 to avoid fog on first frame

        try:
            while True:
                # ── Read next frame ───────────────────────────────────────────
                try:
                    ret, frame = cap.read()
                    if not ret:
                        # End of file — loop back to frame 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            log.warning("Camera %s: cannot read after seek-to-0; sleeping 1 s", camera_id)
                            await asyncio.sleep(1.0)
                            continue
                except Exception as exc:
                    log.warning("Camera %s: frame read error: %s", camera_id, exc)
                    # Hold last valid count — don't broadcast zero
                    await asyncio.sleep(frame_interval)
                    continue

                # ── YOLO inference ────────────────────────────────────────────
                try:
                    t0 = time.perf_counter()
                    results = self.model(frame, conf=0.10, classes=[0], verbose=False)
                    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
                except Exception as exc:
                    log.warning("Camera %s: YOLO inference failed: %s — holding last count", camera_id, exc)
                    await asyncio.sleep(frame_interval)
                    continue

                # ── Compute metrics ───────────────────────────────────────────
                raw_count = len(results[0].boxes) * 2  # ×2 to account for occlusions
                # Hold last valid count to avoid flickering zeroes on bad frames
                if raw_count > 0:
                    last_valid_count = raw_count
                person_count = last_valid_count
                density        = round(person_count / area_m2, 2) if area_m2 > 0 else 0.0
                occupancy_pct  = round(person_count / capacity * 100, 1) if capacity > 0 else 0.0
                congestion_key = classify_congestion(occupancy_pct)
                state          = get_state_config(congestion_key)

                detection_boxes: list[list] = []
                for box in results[0].boxes:
                    try:
                        x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
                        conf = round(float(box.conf[0]), 3)
                        detection_boxes.append([x1, y1, x2, y2, conf])
                    except Exception:
                        pass

                match_minute = self.clock.match_minute
                ts           = time.time()

                payload: dict = {
                    # ── Legacy fields (agents use these exact names) ──────────
                    "zone_id":    zone_id,
                    "label":      zone_label,
                    "occupancy":  person_count,
                    "capacity":   capacity,
                    "utilization": round(person_count / capacity, 3) if capacity else 0.0,
                    "density":    density,
                    "congestion": congestion_key,
                    "area_m2":    area_m2,
                    "match_minute": match_minute,
                    "timestamp":  ts,
                    # ── New / extended fields ─────────────────────────────────
                    "person_count":     person_count,
                    "density_per_m2":   density,
                    "occupancy_pct":    occupancy_pct,
                    "congestion_level": congestion_key,
                    "congestion_color": state["color"],
                    "congestion_icon":  state["icon"],
                    "camera_id":        camera_id,
                    "inference_ms":     inference_ms,
                    "detection_boxes":  detection_boxes,
                    "source":           "camera_inference",
                }

                # ── Publish ───────────────────────────────────────────────────
                # Wrap in the broadcast envelope that agents expect: {"zones": [...]}
                broadcast = {
                    "match_minute": match_minute,
                    "timestamp":    ts,
                    "zones":        [payload],
                    "source":       "camera_inference",
                }
                await self._publish(CH_ZONE_METRICS, broadcast)
                await self._publish(CH_CAMERA_INFERENCE, payload)
                await self._store(f"zone:{zone_id}:latest", payload, ttl=30)

                await asyncio.sleep(frame_interval)

        except asyncio.CancelledError:
            log.info("Camera %s: task cancelled", camera_id)
        finally:
            cap.release()
            log.info("Camera %s: VideoCapture released", camera_id)

    # ── Orchestration ────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Connect to Redis, spawn one task per camera, await graceful shutdown."""
        await self._connect()

        tasks = [
            asyncio.create_task(
                self.process_camera(cam_id, cfg),
                name=f"cam_{cam_id}",
            )
            for cam_id, cfg in CAMERA_FEEDS.items()
        ]
        log.info("Launched %d camera tasks", len(tasks))

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            log.info("CameraInferenceEngine stopped")
            if self.redis:
                await self.redis.aclose()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    engine = CameraInferenceEngine()
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
