from __future__ import annotations

import time

from pydantic import BaseModel, Field


class ZoneUpdate(BaseModel):
    zone_id: str
    density: float          # people per sq meter (0.0 - 6.0 realistic range)
    raw_count: int          # unsmoothed count from current frame
    smoothed_count: float   # EMA-smoothed count
    flow_direction: float = 0.0   # degrees 0-360 (placeholder for Prompt 2)
    flow_magnitude: float = 0.0   # px/frame average (placeholder for Prompt 2)
    gate_in: int = 0        # cumulative IN crossings (placeholder for Prompt 3)
    gate_out: int = 0       # cumulative OUT crossings (placeholder for Prompt 3)
    source: str             # "cv" | "simulator" | "stale_fallback"
    timestamp: float = Field(default_factory=time.time)  # time.time() when generated
    fps: float              # current processing FPS of this zone's feed
