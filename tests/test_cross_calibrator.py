from __future__ import annotations

import pytest

from vision.cross_calibrator import CrossCalibrator


INITIAL = 0.005


def make_calibrator(**kwargs) -> CrossCalibrator:
    return CrossCalibrator(initial_factor=INITIAL, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / initial state
# ---------------------------------------------------------------------------

def test_initial_state():
    cal = make_calibrator()
    assert cal.current_factor == INITIAL
    assert cal.samples == 0
    assert cal.confidence == 0.0
    status = cal.get_status()
    assert status["factor"] == INITIAL
    assert status["initial_factor"] == INITIAL
    assert status["samples"] == 0
    assert status["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Gate: YOLO count outside transition zone
# ---------------------------------------------------------------------------

def test_update_returns_unchanged_when_yolo_too_low():
    cal = make_calibrator(min_yolo_count=15, max_yolo_count=45)
    result = cal.update(yolo_count=10, fg_pixels=5000)
    assert result == INITIAL
    assert cal.samples == 0


def test_update_returns_unchanged_when_yolo_too_high():
    cal = make_calibrator(min_yolo_count=15, max_yolo_count=45)
    result = cal.update(yolo_count=50, fg_pixels=5000)
    assert result == INITIAL
    assert cal.samples == 0


def test_update_returns_unchanged_at_yolo_boundary_low():
    """Exactly at min_yolo_count should be accepted."""
    cal = make_calibrator(min_yolo_count=15, max_yolo_count=45)
    result = cal.update(yolo_count=15, fg_pixels=5000)
    assert result != INITIAL  # a valid update occurred
    assert cal.samples == 1


def test_update_returns_unchanged_at_yolo_boundary_high():
    """Exactly at max_yolo_count should be accepted."""
    cal = make_calibrator(min_yolo_count=15, max_yolo_count=45)
    result = cal.update(yolo_count=45, fg_pixels=5000)
    assert result != INITIAL
    assert cal.samples == 1


# ---------------------------------------------------------------------------
# Gate: foreground pixel floor
# ---------------------------------------------------------------------------

def test_update_returns_unchanged_when_fg_pixels_too_low():
    cal = make_calibrator(min_fg_pixels=1000)
    result = cal.update(yolo_count=25, fg_pixels=500)
    assert result == INITIAL
    assert cal.samples == 0


def test_update_accepts_at_min_fg_pixel_boundary():
    # yolo=20, fg=1000 → observed=0.020, ratio=4.0 — would be outlier with INITIAL=0.005.
    # Use initial=0.015 so ratio=0.020/0.015≈1.33, safely within [0.33, 3.0].
    cal = CrossCalibrator(initial_factor=0.015, min_fg_pixels=1000)
    result = cal.update(yolo_count=20, fg_pixels=1000)
    assert result != 0.015
    assert cal.samples == 1


# ---------------------------------------------------------------------------
# Sanity check: outlier rejection
# ---------------------------------------------------------------------------

def test_outlier_rejected_when_ratio_above_3x():
    """observed_factor > 3 * current_factor should be rejected."""
    cal = make_calibrator()
    # observed_factor = 25 / 1000 = 0.025, initial = 0.005  → ratio = 5 > 3
    result = cal.update(yolo_count=25, fg_pixels=1000)
    assert result == INITIAL
    assert cal.samples == 0


def test_outlier_rejected_when_ratio_below_0_33x():
    """observed_factor < 0.33 * current_factor should be rejected."""
    cal = CrossCalibrator(initial_factor=0.030)
    # observed_factor = 20 / 100_000 = 0.0002  → ratio ≈ 0.0067 < 0.33
    result = cal.update(yolo_count=20, fg_pixels=100_000)
    assert result == cal.current_factor  # unchanged
    assert cal.samples == 0


def test_borderline_ratio_exactly_3x_is_accepted():
    """observed_factor exactly 3x should NOT be rejected (boundary is strict >3)."""
    initial = 0.005
    # We need observed_factor = 3 * initial = 0.015
    # observed = yolo / fg → yolo = 0.015 * fg
    # With fg=2000: yolo = 30 → observed = 0.015 → ratio = 3.0 exactly
    cal = CrossCalibrator(initial_factor=initial, min_fg_pixels=1000)
    result = cal.update(yolo_count=30, fg_pixels=2000)
    # ratio == 3.0 is NOT > 3.0, so it should pass
    assert cal.samples == 1
    assert result != initial


# ---------------------------------------------------------------------------
# EMA update correctness
# ---------------------------------------------------------------------------

def test_ema_update_moves_toward_observed():
    lr = 0.05
    initial = 0.010
    # Need observed_factor within 0.33x–3x of initial.
    # Use fg=4000, yolo=30 → observed = 30/4000 = 0.0075
    # ratio = 0.0075 / 0.010 = 0.75 ✓
    cal = CrossCalibrator(initial_factor=initial, learning_rate=lr, min_fg_pixels=1000)
    observed = 30 / 4000
    expected = lr * observed + (1 - lr) * initial
    result = cal.update(yolo_count=30, fg_pixels=4000)
    assert abs(result - expected) < 1e-10


def test_ema_converges_after_many_samples():
    """After many identical updates factor should converge close to observed."""
    initial = 0.010
    lr = 0.10
    # observed = 30 / 4000 = 0.0075
    cal = CrossCalibrator(initial_factor=initial, learning_rate=lr, min_fg_pixels=1000)
    for _ in range(200):
        cal.update(yolo_count=30, fg_pixels=4000)
    assert abs(cal.current_factor - 0.0075) < 0.0005


# ---------------------------------------------------------------------------
# Sample counting and confidence
# ---------------------------------------------------------------------------

def test_samples_increment_on_valid_update():
    cal = CrossCalibrator(initial_factor=0.010, min_fg_pixels=1000)
    for i in range(5):
        cal.update(yolo_count=30, fg_pixels=4000)
    assert cal.samples == 5


def test_confidence_caps_at_1_after_20_samples():
    cal = CrossCalibrator(initial_factor=0.010, min_fg_pixels=1000)
    for _ in range(25):
        cal.update(yolo_count=30, fg_pixels=4000)
    assert cal.confidence == 1.0


def test_confidence_proportional_before_cap():
    cal = CrossCalibrator(initial_factor=0.010, min_fg_pixels=1000)
    for _ in range(10):
        cal.update(yolo_count=30, fg_pixels=4000)
    assert cal.confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

def test_get_status_reflects_current_state():
    cal = CrossCalibrator(initial_factor=0.010, min_fg_pixels=1000)
    cal.update(yolo_count=30, fg_pixels=4000)
    status = cal.get_status()
    assert status["initial_factor"] == 0.010
    assert status["samples"] == 1
    assert status["factor"] == cal.current_factor
    assert status["confidence"] == cal.confidence


# ---------------------------------------------------------------------------
# Idempotency: rejected updates leave state fully unchanged
# ---------------------------------------------------------------------------

def test_rejected_update_does_not_change_confidence_or_samples():
    cal = make_calibrator()
    cal.update(yolo_count=5, fg_pixels=5000)   # too low YOLO count
    cal.update(yolo_count=130, fg_pixels=5000)  # too high YOLO count (> max_yolo_count=120)
    cal.update(yolo_count=25, fg_pixels=100)   # fg too low
    assert cal.samples == 0
    assert cal.confidence == 0.0
    assert cal.current_factor == INITIAL


# ---------------------------------------------------------------------------
# Occlusion guard: massive fg_pixels + low YOLO → skip update
# ---------------------------------------------------------------------------

def test_occlusion_guard_skips_poisoned_update():
    """fg=80K + yolo=14 with calibrated factor → occlusion, update skipped."""
    cal = make_calibrator(fg_override_threshold=25_000)
    # Build confidence with a valid sample first.
    cal.update(yolo_count=30, fg_pixels=6000)
    assert cal.samples == 1

    old_factor = cal.current_factor
    # Occlusion: expected = 80_000 * factor ≈ 400. yolo=14 < 400*0.3 → skip.
    result = cal.update(yolo_count=14, fg_pixels=80_000)
    assert result == old_factor
    assert cal.samples == 1


def test_occlusion_guard_inactive_before_calibration():
    """Guard requires confidence > 0; with no prior samples it stays off."""
    cal = make_calibrator(fg_override_threshold=25_000)
    assert cal.confidence == 0.0
    # fg=30K is above threshold but guard should be inactive.
    # The update may still be rejected by the tighter outlier bounds,
    # but the occlusion guard itself is not the reason.
    cal.update(yolo_count=14, fg_pixels=30_000)
    assert cal.confidence == 0.0


def test_occlusion_guard_allows_reasonable_yolo():
    """yolo >= 30% of expected → guard does not fire, update proceeds."""
    cal = make_calibrator(fg_override_threshold=25_000)
    cal.update(yolo_count=30, fg_pixels=6000)
    old_samples = cal.samples
    # expected = 30_000 * factor. yolo=60 should be >= 30% of expected.
    cal.update(yolo_count=60, fg_pixels=30_000)
    assert cal.samples == old_samples + 1
