from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CrossCalibrator:
    """Continuously refines MOG2 calibration_factor using YOLO as a teacher.

    During the auto-switch transition zone (min_yolo_count <= YOLO count <=
    max_yolo_count), YOLO detections are reliable ground-truth.  This class
    uses those detections alongside the concurrent MOG2 foreground-pixel count
    to apply a slow EMA update to the calibration_factor.
    """

    def __init__(
        self,
        initial_factor: float,
        learning_rate: float = 0.05,
        # Minimum 10 detections to establish a meaningful pixel-to-person ratio.
        min_yolo_count: int = 10,
        # Raised from 45: YOLO at imgsz=1280 is reliable up to ~120 detections.
        # This allows cross-calibration to learn from medium-to-high density scenes.
        max_yolo_count: int = 120,
        min_fg_pixels: int = 1000,
        fg_override_threshold: int = 25_000,
    ) -> None:
        """
        Args:
            initial_factor: Manual calibration_factor used as the safety baseline.
            learning_rate: EMA weight applied to each new observation (0 < lr < 1).
            min_yolo_count: Minimum YOLO detections required to treat the count
                as ground truth.
            max_yolo_count: Maximum YOLO detections before occlusion makes YOLO
                unreliable.
            min_fg_pixels: Minimum foreground pixels required to avoid
                division-by-near-zero.
            fg_override_threshold: If fg_pixels >= this value and YOLO count
                is suspiciously low, suspect occlusion and skip the update.
        """
        self._initial_factor = initial_factor
        self._current_factor = initial_factor
        self.learning_rate = learning_rate
        self.min_yolo_count = min_yolo_count
        self.max_yolo_count = max_yolo_count
        self.min_fg_pixels = min_fg_pixels
        self.fg_override_threshold = fg_override_threshold

        self._samples: int = 0
        self._confidence: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, yolo_count: int, fg_pixels: int) -> float:
        """Attempt to refine the calibration factor using one frame's data.

        Args:
            yolo_count: Number of people detected by YOLO this frame.
            fg_pixels: Number of foreground pixels reported by MOG2 this frame.

        Returns:
            The current (possibly updated) calibration_factor.
        """
        # Gate 1: YOLO count must be in the reliable transition zone.
        if yolo_count < self.min_yolo_count or yolo_count > self.max_yolo_count:
            return self._current_factor

        # Gate 2: Enough foreground pixels to make division meaningful.
        if fg_pixels < self.min_fg_pixels:
            return self._current_factor

        # Gate 3: Occlusion detection — massive foreground but suspiciously
        # low YOLO count means YOLO is blinded.  Skip to prevent poisoning.
        if fg_pixels >= self.fg_override_threshold and self._confidence > 0:
            expected_count = fg_pixels * self._current_factor
            if expected_count > 0 and yolo_count < expected_count * 0.3:
                logger.info(
                    "CrossCalibrator: occlusion guard — fg_pixels=%d but "
                    "yolo_count=%d (expected ~%.0f) — skipping update",
                    fg_pixels,
                    yolo_count,
                    expected_count,
                )
                return self._current_factor

        observed_factor = yolo_count / fg_pixels

        # Gate 4: reject implausible outliers (>3x or <0.33x current).
        ratio = observed_factor / self._current_factor
        if ratio > 3.0 or ratio < 0.33:
            logger.warning(
                "CrossCalibrator: observed_factor=%.6f is %.2fx current=%.6f — "
                "rejecting outlier",
                observed_factor,
                ratio,
                self._current_factor,
            )
            return self._current_factor

        # Slow EMA update.
        self._current_factor = (
            self.learning_rate * observed_factor
            + (1.0 - self.learning_rate) * self._current_factor
        )
        self._samples += 1
        self._confidence = min(1.0, self._samples / 20.0)

        if self._samples % 10 == 0:
            logger.info(
                "CrossCalibrator: samples=%d factor=%.6f confidence=%.0f%%",
                self._samples,
                self._current_factor,
                self._confidence * 100,
            )

        return self._current_factor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_factor(self) -> float:
        """Current calibration factor (people per foreground pixel)."""
        return self._current_factor

    @property
    def confidence(self) -> float:
        """Calibration confidence in [0.0, 1.0]; reaches 1.0 after 20 samples."""
        return self._confidence

    @property
    def samples(self) -> int:
        """Number of valid calibration samples collected so far."""
        return self._samples

    def get_status(self) -> dict:
        """Return a snapshot of the calibrator's current state.

        Returns:
            Dict with keys: factor, confidence, samples, initial_factor.
        """
        return {
            "factor": self._current_factor,
            "confidence": self._confidence,
            "samples": self._samples,
            "initial_factor": self._initial_factor,
        }