from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_WARMUP_FRAMES = 50


class DenseEstimator:
    """Foreground-pixel density estimator using MOG2 background subtraction.

    Provides a calibrated heuristic crowd count for dense scenes where
    bounding-box detection fundamentally fails due to occlusion.
    During the first _WARMUP_FRAMES calls the background model is still
    building; estimate() returns -1 to signal "not ready yet".
    """

    def __init__(
        self,
        calibration_factor: float = 0.005,
        learning_rate: float = 0.005,
        min_area: int = 500,
    ) -> None:
        """
        Args:
            calibration_factor: People per foreground pixel (tunable per venue).
            learning_rate: MOG2 background model update rate.
            min_area: Minimum contour area to consider (reserved for future
                      contour-based filtering; stored but not applied in the
                      current pixel-count path).
        """
        self.calibration_factor = calibration_factor
        self.learning_rate = learning_rate
        self.min_area = min_area
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self._warmup_frames: int = 0
        self._last_fg_pixels: int = 0

    def estimate(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> int:
        """Estimate crowd count in the given ROI via foreground pixel density.

        Args:
            frame: Full BGR frame from the video source.
            roi: (x, y, w, h) pixel rectangle to analyse.

        Returns:
            Estimated integer count of people, or -1 during the warm-up period
            (first 50 frames) while the background model is still learning.
        """
        x, y, w, h = roi
        roi_frame = frame[y : y + h, x : x + w]

        # Update the background model and extract foreground mask.
        fg_mask = self._bg_subtractor.apply(roi_frame, learningRate=self.learning_rate)

        # Morphological opening removes small noise blobs.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        self._warmup_frames += 1
        if self._warmup_frames <= _WARMUP_FRAMES:
            logger.debug(
                "DenseEstimator warmup frame %d/%d — returning -1",
                self._warmup_frames,
                _WARMUP_FRAMES,
            )
            return -1

        fg_pixels = cv2.countNonZero(cleaned_mask)
        self._last_fg_pixels = fg_pixels
        count = int(fg_pixels * self.calibration_factor)
        logger.debug("DenseEstimator: fg_pixels=%d count=%d", fg_pixels, count)
        return count

    def get_last_fg_pixels(self) -> int:
        """Return the foreground pixel count from the most recent estimate() call."""
        return self._last_fg_pixels
