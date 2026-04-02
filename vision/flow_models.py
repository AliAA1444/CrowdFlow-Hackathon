from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlowResult:
    zone_id: str
    direction_degrees: float  # 0-360, 0=right, 90=down, 180=left, 270=up
    magnitude: float          # average pixel displacement per frame
    is_stagnant: bool         # True if magnitude < threshold
    mean_dx: float            # raw horizontal component
    mean_dy: float            # raw vertical component
