from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateCount:
    gate_in: int
    gate_out: int
    active_tracks: int
