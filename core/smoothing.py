from __future__ import annotations

import logging

logger = logging.getLogger("crowdflow.smoothing")


class EMACounter:
    """Exponential Moving Average counter with spike rejection.

    Protects downstream consumers from violent per-frame fluctuations and
    detection failures (e.g., a single frame where all detections are lost).

    Spike rejection uses a relative threshold (spike_threshold).  A special
    consecutive-rejection override prevents the EMA locking at a stale value
    when the scene legitimately changes: after N_OVERRIDE consecutive rejections
    the raw count is force-accepted and the EMA is reset to it.

    Special case: raw_count == 0 with a significantly positive EMA is always
    treated as a detection failure and rejected regardless of threshold.  It
    does NOT count toward the consecutive-rejection override counter because
    it represents a sensor failure, not a real crowd change.
    """

    N_OVERRIDE = 5  # consecutive rejections before force-accepting

    def __init__(
        self,
        alpha: float = 0.3,
        # FIX Bug #1b: was 0.7 — crowd density can legitimately double between
        # frames (e.g. a group arrives at a gate).  200% still catches true
        # detection failures; zeros are caught by the dedicated zero-guard.
        spike_threshold: float = 2.0,
    ) -> None:
        self.alpha = alpha
        self.spike_threshold = spike_threshold
        self._ema: float | None = None
        self._last_rejected: bool = False
        self._consecutive_rejections: int = 0  # FIX Bug #1a: deadlock detector

    def update(self, raw_count: int) -> float:
        """Update EMA with a new raw count, applying spike rejection.

        Returns:
            The current EMA value (rejected frames leave EMA unchanged).
        """
        # First frame: initialise EMA directly from raw count.
        if self._ema is None:
            self._ema = float(raw_count)
            self._last_rejected = False
            self._consecutive_rejections = 0
            return self._ema

        # Zero-guard: a zero reading with a high EMA is almost certainly a
        # detection failure, not a genuinely empty zone.
        # Does NOT increment _consecutive_rejections — this is structural,
        # not a "the scene changed" signal.
        if raw_count == 0 and self._ema > 10:
            logger.warning(
                "Spike rejected (zero-guard): raw=%d, ema=%.1f", raw_count, self._ema
            )
            self._last_rejected = True
            return self._ema

        # General spike rejection: relative change exceeds threshold.
        relative_change = abs(raw_count - self._ema) / max(self._ema, 1.0)
        if relative_change > self.spike_threshold:
            self._consecutive_rejections += 1
            # FIX Bug #1a: after N_OVERRIDE consecutive rejections the EMA has
            # diverged from reality — force-accept the raw value and reset.
            if self._consecutive_rejections >= self.N_OVERRIDE:
                logger.warning(
                    "Spike filter override: accepting raw=%d after %d consecutive "
                    "rejections (ema was %.1f)",
                    raw_count,
                    self._consecutive_rejections,
                    self._ema,
                )
                self._ema = float(raw_count)
                self._consecutive_rejections = 0
                self._last_rejected = False
                return self._ema
            logger.warning(
                "Spike rejected: raw=%d, ema=%.1f (consecutive=%d)",
                raw_count,
                self._ema,
                self._consecutive_rejections,
            )
            self._last_rejected = True
            return self._ema

        # Standard EMA update.
        self._ema = self.alpha * raw_count + (1.0 - self.alpha) * self._ema
        self._consecutive_rejections = 0
        self._last_rejected = False
        return self._ema

    def reset(self) -> None:
        """Clear all state, returning to pre-first-frame condition."""
        self._ema = None
        self._last_rejected = False
        self._consecutive_rejections = 0

    @property
    def last_was_rejected(self) -> bool:
        """True if the most recent update() call was rejected."""
        return self._last_rejected
