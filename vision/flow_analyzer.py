from __future__ import annotations

import logging

import cv2
import numpy as np

from vision.flow_models import FlowResult

logger = logging.getLogger(__name__)

STAGNATION_THRESHOLD = 0.5  # pixels/frame — below this is considered stagnant


class OpticalFlowAnalyzer:
    """Computes Farnebäck dense optical flow per zone ROI between consecutive frames."""

    def __init__(self, zone_rois: dict[str, tuple[int, int, int, int]]) -> None:
        """
        Args:
            zone_rois: Mapping of zone_id -> (x, y, w, h) pixel rectangle on the camera frame.
        """
        self.zone_rois = zone_rois
        self._prev_gray: np.ndarray | None = None

    def analyze(self, frame: np.ndarray) -> dict[str, FlowResult]:
        """Compute optical flow for each zone ROI.

        Args:
            frame: BGR (or grayscale) frame as a NumPy array (H, W[, C]).

        Returns:
            Mapping of zone_id -> FlowResult. Empty dict on the first call (no previous frame).
        """
        if frame.ndim == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        if self._prev_gray is None:
            self._prev_gray = gray_frame
            return {}

        flow: np.ndarray = cv2.calcOpticalFlowFarneback(
            self._prev_gray,
            gray_frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        frame_h, frame_w = gray_frame.shape[:2]
        results: dict[str, FlowResult] = {}

        for zone_id, (x, y, w, h) in self.zone_rois.items():
            if w <= 0 or h <= 0:
                logger.warning("Zone '%s' has zero-area ROI (w=%d, h=%d) — skipping.", zone_id, w, h)
                continue

            # Clip ROI to frame bounds.
            x1 = max(0, min(x, frame_w))
            y1 = max(0, min(y, frame_h))
            x2 = max(0, min(x + w, frame_w))
            y2 = max(0, min(y + h, frame_h))

            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    "Zone '%s' ROI is entirely outside frame bounds after clipping — skipping.",
                    zone_id,
                )
                continue

            roi_flow = flow[y1:y2, x1:x2]  # shape (roi_h, roi_w, 2)
            mean_dx = float(np.mean(roi_flow[..., 0]))
            mean_dy = float(np.mean(roi_flow[..., 1]))

            direction = float((np.degrees(np.arctan2(mean_dy, mean_dx)) + 360) % 360)
            magnitude = float(np.sqrt(mean_dx ** 2 + mean_dy ** 2))
            is_stagnant = magnitude < STAGNATION_THRESHOLD

            results[zone_id] = FlowResult(
                zone_id=zone_id,
                direction_degrees=direction,
                magnitude=magnitude,
                is_stagnant=is_stagnant,
                mean_dx=mean_dx,
                mean_dy=mean_dy,
            )

        self._prev_gray = gray_frame
        return results
