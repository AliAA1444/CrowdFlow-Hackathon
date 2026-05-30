from __future__ import annotations

"""CrowdFlow — In-memory zone state cache.

Maintains the most-recent ZoneUpdate for every zone so agents can do
cross-zone density comparisons without a Redis round-trip.
"""

from core.schemas import ZoneUpdate

# Adjacency map for the 9-zone config (config/zones.json).
# Used by agents that recommend rerouting from a congested zone to a neighbour.
ADJACENCY: dict[str, list[str]] = {
    "gate_main":       ["concourse_main", "stands_north"],
    "gate_vip":        ["concourse_main", "stands_east"],
    "gate_south":      ["stands_east", "stands_west", "food_court"],
    "concourse_main":  ["gate_main", "gate_vip", "stands_east", "stands_north", "food_court"],
    "stands_east":     ["concourse_main", "gate_vip", "gate_south"],
    "stands_west":     ["stands_north", "gate_south", "food_court"],
    "stands_north":    ["gate_main", "stands_west", "concourse_main"],
    "food_court":      ["concourse_main", "gate_south", "stands_west", "activation_zone"],
    "activation_zone": ["food_court", "gate_main", "gate_south"],
}


class ZoneStateCache:
    """Dict-backed snapshot of the latest ZoneUpdate for each zone."""

    def __init__(self) -> None:
        self._cache: dict[str, ZoneUpdate] = {}

    def update(self, update: ZoneUpdate) -> None:
        """Store (or replace) the latest ZoneUpdate for *update.zone_id*."""
        self._cache[update.zone_id] = update

    def get(self, zone_id: str) -> ZoneUpdate | None:
        """Return the latest ZoneUpdate for *zone_id*, or None if not seen."""
        return self._cache.get(zone_id)

    def get_all(self) -> dict[str, ZoneUpdate]:
        """Return a shallow copy of all cached ZoneUpdates."""
        return dict(self._cache)

    def get_adjacent_zones(self, zone_id: str) -> list[ZoneUpdate]:
        """Return cached ZoneUpdates for all zones adjacent to *zone_id*.

        Zones with no cached update are silently skipped.
        """
        return [
            self._cache[zid]
            for zid in ADJACENCY.get(zone_id, [])
            if zid in self._cache
        ]
