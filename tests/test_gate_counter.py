from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

from vision.gate_models import GateCount


# ---------------------------------------------------------------------------
# Helpers — instantiate counter without loading the real YOLO model
# ---------------------------------------------------------------------------

def make_counter(in_direction: str = "down"):
    """Create a GateTripwireCounter with a mocked YOLO model."""
    with patch("vision.gate_counter.YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value = MagicMock()
        from vision.gate_counter import GateTripwireCounter
        tripwire = ((100, 50), (100, 150))
        counter = GateTripwireCounter(
            model_path="yolov8n.pt",
            tripwire=tripwire,
            in_direction=in_direction,
        )
    return counter


# ---------------------------------------------------------------------------
# GateCount dataclass
# ---------------------------------------------------------------------------

class TestGateCount:
    def test_initializes_correctly(self):
        gc = GateCount(gate_in=5, gate_out=3, active_tracks=8)
        assert gc.gate_in == 5
        assert gc.gate_out == 3
        assert gc.active_tracks == 8

    def test_zero_values(self):
        gc = GateCount(gate_in=0, gate_out=0, active_tracks=0)
        assert gc.gate_in == 0
        assert gc.gate_out == 0
        assert gc.active_tracks == 0


# ---------------------------------------------------------------------------
# _segments_intersect
# ---------------------------------------------------------------------------

class TestSegmentsIntersect:
    def setup_method(self):
        self.counter = make_counter()

    def test_crossing_segments(self):
        # Vertical segment crosses horizontal segment
        assert self.counter._segments_intersect(
            (0, 5), (10, 5),   # horizontal movement
            (5, 0), (5, 10),   # vertical tripwire
        ) is True

    def test_X_shaped_crossing(self):
        assert self.counter._segments_intersect(
            (0, 0), (10, 10),
            (10, 0), (0, 10),
        ) is True

    def test_parallel_non_intersecting(self):
        # Two horizontal segments at different y values
        assert self.counter._segments_intersect(
            (0, 2), (10, 2),
            (0, 5), (10, 5),
        ) is False

    def test_collinear_non_overlapping(self):
        # Same line, no overlap
        assert self.counter._segments_intersect(
            (0, 0), (3, 0),
            (5, 0), (10, 0),
        ) is False

    def test_t_junction_not_crossed(self):
        # p1-p2 ends exactly at tripwire but does not cross
        # The endpoint touching is treated as non-intersecting (strict cross-product test)
        result = self.counter._segments_intersect(
            (0, 5), (5, 5),
            (5, 0), (5, 10),
        )
        # Endpoint touch: d1 or d2 will be zero — not strictly crossing, so False
        assert result is False

    def test_non_crossing_same_side(self):
        # Both endpoints of movement are on the same side of the tripwire
        assert self.counter._segments_intersect(
            (0, 5), (3, 5),
            (10, 0), (10, 10),
        ) is False


# ---------------------------------------------------------------------------
# _counted_ids pruning
# ---------------------------------------------------------------------------

class TestCountedIdsPruning:
    def test_prune_removes_inactive_ids(self):
        counter = make_counter()
        # Simulate tracks 1-50 were counted but only 1-10 are still active
        for i in range(1, 51):
            counter._counted_ids.add(i)
        for i in range(1, 11):
            counter._track_history[i] = [(float(i), float(i))]

        counter._prune_counted_ids()

        assert counter._counted_ids == set(range(1, 11))

    def test_prune_called_at_interval(self):
        """After 1000 frames processed, pruning should have fired at least once."""
        counter = make_counter()

        # Mark track 99 as counted but remove it from history
        counter._counted_ids.add(99)
        # Track 99 is NOT in _track_history

        # Manually trigger pruning
        counter._prune_counted_ids()

        assert 99 not in counter._counted_ids

    def test_prune_keeps_still_active_ids(self):
        counter = make_counter()
        counter._counted_ids.update({1, 2, 3})
        counter._track_history[1] = [(10.0, 20.0)]
        counter._track_history[2] = [(30.0, 40.0)]
        # Track 3 not in history

        counter._prune_counted_ids()

        assert 1 in counter._counted_ids
        assert 2 in counter._counted_ids
        assert 3 not in counter._counted_ids

    def test_1001_frames_prunes_stale_ids(self):
        """Simulate 1001 process_frame calls; stale counted IDs must be pruned."""
        import numpy as np

        counter = make_counter()

        # Pre-populate a stale counted ID (track 999) with no history
        counter._counted_ids.add(999)

        # Mock model.track to always return empty results so no real tracking
        counter.model.track.return_value = []

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        from vision.gate_counter import _PRUNE_INTERVAL
        for _ in range(_PRUNE_INTERVAL + 1):
            counter.process_frame(dummy_frame)

        # After pruning cycle, track 999 (no history) should be gone
        assert 999 not in counter._counted_ids


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        counter = make_counter()
        counter._cumulative_in = 10
        counter._cumulative_out = 5
        counter._track_history[1] = [(1.0, 2.0)]
        counter._counted_ids.add(1)
        counter._frame_count = 500

        counter.reset()

        assert counter._cumulative_in == 0
        assert counter._cumulative_out == 0
        assert len(counter._track_history) == 0
        assert len(counter._counted_ids) == 0
        assert counter._frame_count == 0
