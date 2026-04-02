from __future__ import annotations

import logging

logger = logging.getLogger("crowdflow.smoothing")


class EMACounter:
    """Exponential Moving Average counter with spike rejection.

    Protects downstream consumers from violent per-frame fluctuations and
    detection failures (e.g., a single frame where all detections are lost).
    """

    def __init__(self, alpha: float = 0.3, spike_threshold: float = 0.7) -> None:
        self.alpha = alpha
        self.spike_threshold = spike_threshold
        self._ema: float | None = None
        self._last_rejected: bool = False

    def update(self, raw_count: int) -> float:
        """Update EMA with a new raw count, applying spike rejection.

        Returns:
            The current EMA value (rejected frames leave EMA unchanged).
        """
        # First frame: initialise EMA directly from raw count.
        if self._ema is None:
            self._ema = float(raw_count)
            self._last_rejected = False
            return self._ema

        # Zero-guard: a zero reading with a high EMA is almost certainly a
        # detection failure, not a genuinely empty zone.
        if raw_count == 0 and self._ema > 10:
            logger.warning(
                "Spike rejected (zero-guard): raw=%d, ema=%.1f", raw_count, self._ema
            )
            self._last_rejected = True
            return self._ema

        # General spike rejection: relative change exceeds threshold.
        relative_change = abs(raw_count - self._ema) / max(self._ema, 1.0)
        if relative_change > self.spike_threshold:
            logger.warning(
                "Spike rejected: raw=%d, ema=%.1f", raw_count, self._ema
            )
            self._last_rejected = True
            return self._ema

        # Standard EMA update.
        self._ema = self.alpha * raw_count + (1.0 - self.alpha) * self._ema
        self._last_rejected = False
        return self._ema

    def reset(self) -> None:
        """Clear all state, returning to pre-first-frame condition."""
        self._ema = None
        self._last_rejected = False

    @property
    def last_was_rejected(self) -> bool:
        """True if the most recent update() call was rejected."""
        return self._last_rejected
