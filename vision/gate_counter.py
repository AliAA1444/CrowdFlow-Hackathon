from __future__ import annotations

from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from vision.gate_models import GateCount

_DIRECTION_LABELS = {"up", "down", "left", "right"}
_HISTORY_MAX = 30
_PRUNE_INTERVAL = 1000


class GateTripwireCounter:
    """Counts people crossing a virtual tripwire line using ByteTrack."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tripwire: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (100, 0)),
        in_direction: str = "down",
        # CV TUNING: lower confidence catches partially visible people at gate edges
        confidence_threshold: float = 0.15,
    ) -> None:
        if in_direction not in _DIRECTION_LABELS:
            raise ValueError(f"in_direction must be one of {_DIRECTION_LABELS}")

        self.model = YOLO(model_path)
        self.tripwire = tripwire
        self.in_direction = in_direction
        self.confidence_threshold = confidence_threshold

        self._cumulative_in: int = 0
        self._cumulative_out: int = 0
        self._track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._counted_ids: set[int] = set()
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> GateCount:
        self._frame_count += 1

        if self._frame_count % _PRUNE_INTERVAL == 0:
            self._prune_counted_ids()

        # CV TUNING: lower confidence catches partially visible people at gate edges
        # imgsz=1280 improves detection of people further from camera
        # persist=True is critical — do not remove (maintains ByteTrack state)
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            classes=[0],
            verbose=False,
            imgsz=1280,
        )

        if results is None or len(results) == 0:
            return GateCount(
                gate_in=self._cumulative_in,
                gate_out=self._cumulative_out,
                active_tracks=len(self._track_history),
            )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return GateCount(
                gate_in=self._cumulative_in,
                gate_out=self._cumulative_out,
                active_tracks=len(self._track_history),
            )

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            history = self._track_history[track_id]
            history.append((cx, cy))
            if len(history) > _HISTORY_MAX:
                history.pop(0)

        tw_start, tw_end = self.tripwire
        for track_id, history in self._track_history.items():
            if len(history) < 2 or track_id in self._counted_ids:
                continue

            prev = history[-2]
            curr = history[-1]

            if self._segments_intersect(prev, curr, tw_start, tw_end):
                direction = self._crossing_direction(prev, curr, tw_start, tw_end)
                if direction == "in":
                    self._cumulative_in += 1
                else:
                    self._cumulative_out += 1
                self._counted_ids.add(track_id)

        return GateCount(
            gate_in=self._cumulative_in,
            gate_out=self._cumulative_out,
            active_tracks=len(self._track_history),
        )

    def draw_tripwire(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        p1 = self.tripwire[0]
        p2 = self.tripwire[1]
        cv2.line(out, p1, p2, (0, 255, 255), 2)
        cv2.putText(out, f"IN: {self._cumulative_in}", (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"OUT: {self._cumulative_out}", (p1[0], p1[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out

    def reset(self) -> None:
        self._cumulative_in = 0
        self._cumulative_out = 0
        self._track_history.clear()
        self._counted_ids.clear()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        """2D cross product of vectors OA and OB."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _segments_intersect(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> bool:
        """Return True if segment p1-p2 intersects segment p3-p4."""
        d1 = self._cross(p3, p4, p1)
        d2 = self._cross(p3, p4, p2)
        d3 = self._cross(p1, p2, p3)
        d4 = self._cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # Collinear cases — treat as non-intersecting for counting purposes
        return False

    def _crossing_direction(
        self,
        movement_start: tuple[float, float],
        movement_end: tuple[float, float],
        tripwire_start: tuple[float, float],
        tripwire_end: tuple[float, float],
    ) -> str:
        """
        Determine whether the crossing is IN or OUT.

        The cross product of the tripwire vector and the movement vector gives
        the signed area. A positive value means the movement goes to the left of
        the tripwire direction; negative means right.

        Mapping to screen directions (origin top-left, y increases downward):
          tripwire horizontal (left→right): positive cross → movement goes UP
          tripwire vertical   (top→bottom): positive cross → movement goes LEFT
        """
        tw_dx = tripwire_end[0] - tripwire_start[0]
        tw_dy = tripwire_end[1] - tripwire_start[1]
        mv_dx = movement_end[0] - movement_start[0]
        mv_dy = movement_end[1] - movement_start[1]

        # cross = tw × mv
        cross = tw_dx * mv_dy - tw_dy * mv_dx

        # Determine the "positive-cross" side in screen-space terms
        if self.in_direction == "down":
            return "in" if cross < 0 else "out"
        elif self.in_direction == "up":
            return "in" if cross > 0 else "out"
        elif self.in_direction == "right":
            return "in" if cross > 0 else "out"
        else:  # left
            return "in" if cross < 0 else "out"

    # ------------------------------------------------------------------
    # Internal maintenance
    # ------------------------------------------------------------------

    def _prune_counted_ids(self) -> None:
        """Remove counted IDs that are no longer being tracked."""
        active = set(self._track_history.keys())
        self._counted_ids &= active
