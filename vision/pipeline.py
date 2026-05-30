from __future__ import annotations

import asyncio
import logging
import time

import cv2
import numpy as np
import redis.asyncio as aioredis
from ultralytics import YOLO

from core.redis_publisher import ZonePublisher
from vision.cross_calibrator import CrossCalibrator
from vision.density_estimator import DenseEstimator
from vision.flow_analyzer import OpticalFlowAnalyzer
from vision.gate_counter import GateTripwireCounter

logger = logging.getLogger(__name__)


# HARDWARE PROTECTION: Only run heavy CV inference every Nth frame.
# cap.read() runs every frame to keep video clock synchronized,
# but YOLO/MOG2/ByteTrack only execute on sampled frames.
# This reduces CPU load by ~70% (e.g., process_every=3 means skip 2 of 3 frames).
PROCESS_EVERY_N_FRAMES = 3


class VisionPipeline:
    """Unified vision pipeline with automatic YOLO / dense-estimation mode switching.

    Architecture
    ------------
    - One YOLO model (shared) for general-zone detection.
    - One DenseEstimator per general zone (each holds independent MOG2 state).
    - One GateTripwireCounter per gate zone (uses its own internal YOLO tracker).
    - One OpticalFlowAnalyzer shared across all zones.
    - One ZonePublisher per zone for Redis I/O.

    Mode switching (general zones)
    ------------------------------
    If YOLO detects >= switch_threshold people the zone is considered dense and
    the DenseEstimator takes over as the count source ("cv_dense").  Below that
    threshold YOLO remains the count source ("cv_yolo").  During DenseEstimator
    warmup (returns -1) the pipeline falls back to the YOLO count automatically.
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        zone_config: dict | None = None,
        # Auto-switch to MOG2 dense estimator when YOLO detects >= 100 people.
        # At imgsz=1280, YOLO is reliable up to ~100-120 detections.
        # Below 100: trust YOLO (it's more accurate for sparse/medium density).
        # Above 100: YOLO starts missing people due to occlusion, switch to MOG2.
        switch_threshold: int = 100,
        fg_override_threshold: int = 25_000,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        if zone_config is None:
            zone_config = {"zones": {}}

        self.zone_config = zone_config
        self.switch_threshold = switch_threshold
        self.fg_override_threshold = fg_override_threshold

        # Concurrency semaphore: at most 3 zones run heavy CV inference simultaneously.
        # On a MacBook Air (8 cores), this prevents CPU saturation across 9 feeds.
        # cap.read() is NOT gated — all feeds keep reading frames for clock sync.
        self._inference_semaphore = asyncio.Semaphore(3)

        # Single YOLO instance shared across all general-zone detection calls.
        self.model = YOLO(yolo_model_path)

        redis_client = aioredis.from_url(redis_url)

        # One OpticalFlowAnalyzer per zone so each feed's _prev_gray state is
        # completely isolated.  A shared instance crashes when concurrent feeds
        # have different frame resolutions (OpenCV requires prev/next same size).
        self.flow_analyzers: dict[str, OpticalFlowAnalyzer] = {}
        for zone_id, cfg in zone_config["zones"].items():
            r = cfg["roi"]
            self.flow_analyzers[zone_id] = OpticalFlowAnalyzer(
                {zone_id: (r["x"], r["y"], r["w"], r["h"])}
            )

        # Per-zone components, keyed by zone_id.
        self.publishers: dict[str, ZonePublisher] = {}
        self.dense_estimators: dict[str, DenseEstimator] = {}
        self.cross_calibrators: dict[str, CrossCalibrator] = {}
        self.gate_counters: dict[str, GateTripwireCounter] = {}
        self._zone_fps: dict[str, float] = {}

        for zone_id, cfg in zone_config["zones"].items():
            zone_type = cfg["type"]
            self.publishers[zone_id] = ZonePublisher(redis_client, zone_id)
            self._zone_fps[zone_id] = 0.0

            if zone_type == "general":
                cal = float(cfg.get("calibration_factor", 0.005))
                self.dense_estimators[zone_id] = DenseEstimator(
                    calibration_factor=cal
                )
                self.cross_calibrators[zone_id] = CrossCalibrator(
                    initial_factor=cal,
                    fg_override_threshold=fg_override_threshold,
                )

            elif zone_type == "gate":
                tw = cfg["tripwire"]
                tripwire = (tuple(tw["p1"]), tuple(tw["p2"]))
                self.gate_counters[zone_id] = GateTripwireCounter(
                    model_path=yolo_model_path,
                    tripwire=tripwire,
                    in_direction=cfg["in_direction"],
                )
                cal = float(cfg.get("calibration_factor", 0.0025))
                self.dense_estimators[zone_id] = DenseEstimator(
                    calibration_factor=cal
                )
                self.cross_calibrators[zone_id] = CrossCalibrator(
                    initial_factor=cal,
                    fg_override_threshold=fg_override_threshold,
                )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def process_frame(
        self,
        zone_id: str,
        frame: np.ndarray,
        zone_type: str,
    ) -> None:
        """Process a single frame for *zone_id* and publish to Redis.

        Args:
            zone_id: Zone identifier (must exist in zone_config).
            frame: BGR frame as a NumPy array.
            zone_type: "general" or "gate".
        """
        cfg = self.zone_config["zones"][zone_id]
        area_sqm = float(cfg["area_sqm"])
        r = cfg["roi"]
        roi = (r["x"], r["y"], r["w"], r["h"])
        fps = self._zone_fps.get(zone_id, 0.0)
        publisher = self.publishers[zone_id]

        if zone_type == "gate":
            # Tripwire crossing tracking (cumulative in/out).
            gate_count = self.gate_counters[zone_id].process_frame(frame)

            # Instantaneous detection count for the gate zone.
            results = self.model(frame, conf=0.15, classes=[0], verbose=False, imgsz=1280)
            yolo_count = len(results[0].boxes)

            dense_est = self.dense_estimators[zone_id]
            cross_cal = self.cross_calibrators[zone_id]

            # Always run MOG2 to keep background model warm and update fg_pixels.
            # Without this, get_last_fg_pixels() stays 0 forever (chicken-and-egg).
            dense_count = dense_est.estimate(frame, roi)
            last_fg = dense_est.get_last_fg_pixels()

            updated_factor = cross_cal.update(yolo_count, last_fg)
            dense_est.calibration_factor = updated_factor

            # Mode switching: use MOG2 when YOLO saturates or FG override fires.
            fg_override = last_fg >= self.fg_override_threshold
            use_dense = yolo_count >= self.switch_threshold or fg_override

            if fg_override and yolo_count < self.switch_threshold:
                logger.info(
                    "Gate %s: fg_pixel override — fg=%d, yolo=%d "
                    "— forcing dense mode",
                    zone_id,
                    last_fg,
                    yolo_count,
                )

            if use_dense and dense_count != -1:
                raw_count = dense_count
                if cross_cal.confidence > 0:
                    source = (
                        f"cv_dense:cal@{updated_factor:.6f}:"
                        f"{int(cross_cal.confidence * 100)}%"
                    )
                else:
                    source = "cv_dense"
            else:
                raw_count = yolo_count
                source = "cv_yolo"

            # Coverage-ratio extrapolation: scale count to full sector if configured.
            sector_sqm = float(cfg.get("sector_total_sqm", area_sqm))
            coverage_ratio = sector_sqm / max(area_sqm, 1.0)
            if coverage_ratio > 1.0:
                raw_count = int(raw_count * coverage_ratio)
                source = f"{source}:x{coverage_ratio:.0f}"

            await publisher.publish(
                raw_count=raw_count,
                source=source,
                fps=fps,
                zone_area_sqm=sector_sqm,
                gate_in=gate_count.gate_in,
                gate_out=gate_count.gate_out,
            )

        elif zone_type == "general":
            results = self.model(frame, conf=0.15, classes=[0], verbose=False, imgsz=1280)
            yolo_count = len(results[0].boxes)

            dense_est = self.dense_estimators[zone_id]
            cross_cal = self.cross_calibrators[zone_id]

            # Always run MOG2 to keep background model warm and update fg_pixels.
            dense_count = dense_est.estimate(frame, roi)
            last_fg = dense_est.get_last_fg_pixels()

            updated_factor = cross_cal.update(yolo_count, last_fg)
            dense_est.calibration_factor = updated_factor

            fg_override = last_fg >= self.fg_override_threshold
            use_dense = yolo_count >= self.switch_threshold or fg_override

            if fg_override and yolo_count < self.switch_threshold:
                logger.info(
                    "Zone %s: fg_pixel override — fg=%d, yolo=%d "
                    "— forcing dense mode",
                    zone_id,
                    last_fg,
                    yolo_count,
                )

            if use_dense and dense_count != -1:
                raw_count = dense_count
                if cross_cal.confidence > 0:
                    source = (
                        f"cv_dense:cal@{updated_factor:.6f}:"
                        f"{int(cross_cal.confidence * 100)}%"
                    )
                else:
                    source = "cv_dense"
            else:
                raw_count = yolo_count
                source = "cv_yolo"

            # Coverage-ratio extrapolation: camera covers area_sqm of sector_total_sqm.
            # Density stays consistent: (count * ratio) / (area * ratio) == count / area.
            # The displayed count scales to the full sector; density drives agent logic.
            sector_sqm = float(cfg.get("sector_total_sqm", area_sqm))
            coverage_ratio = sector_sqm / max(area_sqm, 1.0)
            if coverage_ratio > 1.0:
                raw_count = int(raw_count * coverage_ratio)
                source = f"{source}:x{coverage_ratio:.0f}"

            await publisher.publish(
                raw_count=raw_count,
                source=source,
                fps=fps,
                zone_area_sqm=sector_sqm,
            )

        # Optical flow runs on every frame regardless of mode.
        flow_results = self.flow_analyzers[zone_id].analyze(frame)
        if zone_id in flow_results:
            flow = flow_results[zone_id]
            logger.debug(
                "Zone %s flow: dir=%.1f° mag=%.2f px/frame stagnant=%s",
                zone_id,
                flow.direction_degrees,
                flow.magnitude,
                flow.is_stagnant,
            )

    # ------------------------------------------------------------------
    # Feed management
    # ------------------------------------------------------------------

    async def run_feed(
        self,
        zone_id: str,
        video_source: str,
        zone_type: str,
    ) -> None:
        """Read frames from *video_source* indefinitely and process each one.

        Video files are looped endlessly (simulating a live camera) — when
        cap.read() signals end-of-file the feed is rewound to frame 0.  Only
        if reading still fails after the rewind is the zone marked stale and
        this coroutine exits.

        cap.read() is wrapped in asyncio.to_thread() to keep the event loop
        unblocked while OpenCV performs I/O.
        """
        cap = cv2.VideoCapture(video_source)
        publisher = self.publishers[zone_id]
        prev_time = time.perf_counter()

        logger.info("Zone %s: starting feed from '%s'", zone_id, video_source)

        frame_index = 0

        while True:
            ret, frame = await asyncio.to_thread(cap.read)

            if not ret:
                # End of video file — rewind and continue like a live camera.
                logger.debug(
                    "Zone %s: end of '%s', rewinding to frame 0",
                    zone_id,
                    video_source,
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    logger.error(
                        "Zone %s: feed '%s' failed after rewind — marking stale",
                        zone_id,
                        video_source,
                    )
                    await publisher.mark_stale(
                        f"Feed '{video_source}' failed after rewind"
                    )
                    break

            frame_index += 1

            # Always read frames to keep video in sync, but only run heavy
            # CV inference (YOLO/MOG2/ByteTrack) every Nth frame.
            if frame_index % PROCESS_EVERY_N_FRAMES != 0:
                continue

            # Update per-zone FPS estimate (based on processed frames only).
            now = time.perf_counter()
            elapsed = now - prev_time
            self._zone_fps[zone_id] = 1.0 / elapsed if elapsed > 0 else 0.0
            prev_time = now

            # Semaphore: at most 3 zones run heavy CV inference at the same time.
            async with self._inference_semaphore:
                await self.process_frame(zone_id, frame, zone_type)

        cap.release()

    async def run_all_feeds(self, feed_config: dict[str, str]) -> None:
        """Launch one feed task per zone and run them all concurrently.

        A crash in one zone's feed is caught and logged; other zones continue
        running unaffected.

        Args:
            feed_config: Mapping of zone_id -> video_source path / URL.
        """
        async def _safe_run(zone_id: str, video_source: str) -> None:
            zone_type = self.zone_config["zones"].get(zone_id, {}).get(
                "type", "general"
            )
            try:
                await self.run_feed(zone_id, video_source, zone_type)
            except Exception:
                logger.exception(
                    "Zone %s feed raised an unhandled exception — isolating failure",
                    zone_id,
                )

        logger.info(
            "Pipeline: %d zones, process every %d frames, max 3 concurrent inference",
            len(feed_config),
            PROCESS_EVERY_N_FRAMES,
        )
        tasks = [
            _safe_run(zone_id, src) for zone_id, src in feed_config.items()
        ]
        await asyncio.gather(*tasks)
