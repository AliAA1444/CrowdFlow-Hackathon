from __future__ import annotations

import sys
import os

# Ensure project root is on the path when running from any directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.smoothing import EMACounter


# ---------------------------------------------------------------------------
# First-frame initialisation
# ---------------------------------------------------------------------------

def test_first_frame_init() -> None:
    ema = EMACounter()
    result = ema.update(100)
    assert result == 100.0
    assert ema.last_was_rejected is False


def test_first_frame_zero() -> None:
    ema = EMACounter()
    result = ema.update(0)
    assert result == 0.0
    assert ema.last_was_rejected is False


# ---------------------------------------------------------------------------
# EMA convergence
# ---------------------------------------------------------------------------

def test_ema_convergence() -> None:
    """Feeding a constant value should converge to that value."""
    ema = EMACounter(alpha=0.3)
    for _ in range(20):
        result = ema.update(100)
    # After 20 frames at alpha=0.3, EMA should be within 1% of 100.
    assert abs(result - 100.0) < 1.0, f"EMA did not converge: {result}"
    assert ema.last_was_rejected is False


# ---------------------------------------------------------------------------
# Spike rejection
# ---------------------------------------------------------------------------

def test_spike_rejection_zero_with_high_ema() -> None:
    """A zero count when EMA > 10 must be rejected (detection failure)."""
    ema = EMACounter(alpha=0.3)
    # Prime the EMA to 100.
    ema.update(100)
    prev_ema = ema.update(100)  # stabilise

    result = ema.update(0)

    assert result == prev_ema, f"Zero spike should be rejected; got {result}"
    assert ema.last_was_rejected is True


def test_spike_rejection_large_jump() -> None:
    """A sudden doubling (relative change = 1.0 > 0.7) must be rejected."""
    ema = EMACounter(alpha=0.3, spike_threshold=0.7)
    prev_ema = ema.update(100)

    result = ema.update(200)  # relative change = 100/100 = 1.0 > 0.7

    assert result == prev_ema, f"Large jump should be rejected; got {result}"
    assert ema.last_was_rejected is True


def test_spike_not_rejected_when_small() -> None:
    """A moderate increase (relative change = 0.1 < 0.7) must be accepted."""
    ema = EMACounter(alpha=0.3, spike_threshold=0.7)
    ema.update(100)

    result = ema.update(110)  # relative change = 10/100 = 0.1

    expected = 0.3 * 110 + 0.7 * 100
    assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"
    assert ema.last_was_rejected is False


# ---------------------------------------------------------------------------
# Gradual decline is accepted
# ---------------------------------------------------------------------------

def test_gradual_decline_accepted() -> None:
    """A gradual step-down sequence should never trigger spike rejection."""
    ema = EMACounter(alpha=0.3, spike_threshold=0.7)
    values = [100, 90, 80, 70, 60]

    for v in values:
        ema.update(v)
        assert ema.last_was_rejected is False, (
            f"Value {v} was incorrectly rejected; gradual decline should be accepted"
        )


# ---------------------------------------------------------------------------
# last_was_rejected property
# ---------------------------------------------------------------------------

def test_last_was_rejected_toggles() -> None:
    """last_was_rejected should flip between True/False as frames are accepted/rejected."""
    ema = EMACounter(alpha=0.3, spike_threshold=0.7)
    ema.update(100)

    # Accepted update
    ema.update(110)
    assert ema.last_was_rejected is False

    # Rejected update (spike)
    ema.update(300)
    assert ema.last_was_rejected is True

    # Accepted update again
    ema.update(112)
    assert ema.last_was_rejected is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state() -> None:
    """After reset(), the next update should behave like the first frame."""
    ema = EMACounter(alpha=0.3)
    ema.update(100)
    ema.update(100)

    ema.reset()

    # First frame after reset: EMA should initialise to raw_count exactly.
    result = ema.update(50)
    assert result == 50.0
    assert ema.last_was_rejected is False


def test_reset_zero_guard_disabled_after_reset() -> None:
    """After reset(), update(0) should initialise EMA to 0, not be rejected."""
    ema = EMACounter(alpha=0.3)
    ema.update(100)
    ema.reset()

    result = ema.update(0)
    assert result == 0.0
    assert ema.last_was_rejected is False
