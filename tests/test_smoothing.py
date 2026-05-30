from __future__ import annotations

import sys
import os

import pytest

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


# ---------------------------------------------------------------------------
# Consecutive-rejection override (Bug #1a fix)
# ---------------------------------------------------------------------------

def test_consecutive_rejection_override() -> None:
    """After N_OVERRIDE consecutive rejections the EMA is force-reset to raw."""
    ema = EMACounter(alpha=0.3, spike_threshold=2.0)
    # Seed a low initial EMA (simulates sparse first frame).
    ema.update(37)  # EMA = 37

    # Frames 1-4: relative change = (250-37)/37 ≈ 5.76 > 2.0 → rejected each time.
    for i in range(4):
        r = ema.update(250)
        assert ema.last_was_rejected is True, (
            f"Frame {i + 1}: expected rejection but got ema={r}"
        )
        assert r == pytest.approx(37.0), (
            f"EMA should stay at 37 during rejection; got {r}"
        )

    # Frame 5: 5th consecutive rejection → force-override.
    r5 = ema.update(250)
    assert r5 == pytest.approx(250.0), (
        f"5th consecutive rejection should force-accept raw=250; got {r5}"
    )
    assert ema.last_was_rejected is False

    # EMA must stay near 250 on subsequent identical frames.
    for _ in range(5):
        result = ema.update(250)
    assert result > 200, (
        f"Post-override EMA should converge to 250; got {result}"
    )


def test_zero_still_rejected_with_high_threshold() -> None:
    """Zero count is always rejected when EMA > 10, even with threshold=2.0.

    (abs(0 - 100) / 100 = 1.0 which is < 2.0, so without the dedicated
    zero-guard this would be accepted.  The zero-guard must fire first.)
    """
    ema = EMACounter(alpha=0.3, spike_threshold=2.0)
    ema.update(100)
    ema.update(100)  # EMA ≈ 100

    val = ema.update(0)
    assert val > 50, f"Zero with high EMA should be rejected; ema dropped to {val}"
    assert ema.last_was_rejected is True

    # The zero-guard must NOT count toward the consecutive-rejection counter.
    assert ema._consecutive_rejections == 0, (
        "Zero-guard rejection must not increment _consecutive_rejections"
    )
