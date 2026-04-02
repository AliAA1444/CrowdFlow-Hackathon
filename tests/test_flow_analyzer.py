from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from vision.flow_analyzer import OpticalFlowAnalyzer


def _black_frame(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _shifted_right(frame: np.ndarray, px: int = 5) -> np.ndarray:
    """Return frame shifted `px` pixels to the right (left edge filled with black)."""
    shifted = np.zeros_like(frame)
    shifted[:, px:] = frame[:, :-px]
    return shifted


# ---------------------------------------------------------------------------
# Rightward motion -> direction ~0 degrees
# ---------------------------------------------------------------------------

def test_rightward_motion_direction() -> None:
    zone_rois = {"zone_a": (0, 0, 100, 100)}
    analyzer = OpticalFlowAnalyzer(zone_rois)

    # Use a non-uniform frame so Farnebäck has texture to track.
    rng = np.random.default_rng(42)
    frame1 = rng.integers(50, 200, (100, 100, 3), dtype=np.uint8)
    frame2 = _shifted_right(frame1, px=5)

    # First call: no previous frame, returns empty.
    result = analyzer.analyze(frame1)
    assert result == {}

    # Second call: should detect rightward motion.
    result = analyzer.analyze(frame2)
    assert "zone_a" in result

    fr = result["zone_a"]
    assert fr.magnitude > 0, "Magnitude should be > 0 for shifted frame"
    # Direction 0 = right. Allow ±30 degrees tolerance for optical flow approximation.
    assert abs(fr.direction_degrees) < 30 or fr.direction_degrees > 330, (
        f"Expected direction ~0 (rightward), got {fr.direction_degrees:.1f}"
    )
    assert fr.is_stagnant is False


# ---------------------------------------------------------------------------
# Identical frames -> stagnation
# ---------------------------------------------------------------------------

def test_stagnation_identical_frames() -> None:
    zone_rois = {"zone_b": (0, 0, 100, 100)}
    analyzer = OpticalFlowAnalyzer(zone_rois)

    frame = _black_frame()
    analyzer.analyze(frame)
    result = analyzer.analyze(frame)

    assert "zone_b" in result
    assert result["zone_b"].is_stagnant is True
    assert result["zone_b"].magnitude < 0.5


# ---------------------------------------------------------------------------
# ROI clipped to frame bounds -> no crash
# ---------------------------------------------------------------------------

def test_roi_clipping_no_crash() -> None:
    # ROI extends well beyond the 100x100 frame.
    zone_rois = {"zone_c": (80, 80, 200, 200)}
    analyzer = OpticalFlowAnalyzer(zone_rois)

    frame1 = _black_frame(100, 100)
    frame2 = _black_frame(100, 100)
    frame2[85:100, 85:100] = 128  # add some content in the visible part

    analyzer.analyze(frame1)
    result = analyzer.analyze(frame2)
    # Should not raise; zone_c should appear (clipped area is non-empty).
    assert "zone_c" in result


# ---------------------------------------------------------------------------
# Zero-area ROI -> gracefully skipped
# ---------------------------------------------------------------------------

def test_zero_area_roi_skipped() -> None:
    zone_rois = {
        "zero_w": (10, 10, 0, 50),
        "zero_h": (10, 10, 50, 0),
    }
    analyzer = OpticalFlowAnalyzer(zone_rois)

    frame = _black_frame()
    analyzer.analyze(frame)
    result = analyzer.analyze(frame)

    assert "zero_w" not in result
    assert "zero_h" not in result


# ---------------------------------------------------------------------------
# First call always returns empty dict
# ---------------------------------------------------------------------------

def test_first_call_returns_empty() -> None:
    analyzer = OpticalFlowAnalyzer({"z": (0, 0, 50, 50)})
    result = analyzer.analyze(_black_frame())
    assert result == {}
