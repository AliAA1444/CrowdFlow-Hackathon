from __future__ import annotations

import json


def load_zone_rois(config_path: str) -> dict[str, tuple[int, int, int, int]]:
    """Load zone ROI definitions from a JSON config file.

    Expected format::

        {
            "gate_a": {"x": 0, "y": 0, "w": 320, "h": 240},
            ...
        }

    Returns a mapping of zone_id -> (x, y, w, h).
    Raises ValueError if any coordinate value is not a non-negative integer.
    """
    with open(config_path, "r") as f:
        raw: dict = json.load(f)

    result: dict[str, tuple[int, int, int, int]] = {}
    for zone_id, coords in raw.items():
        for key in ("x", "y", "w", "h"):
            val = coords.get(key)
            if not isinstance(val, int) or val < 0:
                raise ValueError(
                    f"Zone '{zone_id}': '{key}' must be a non-negative integer, got {val!r}"
                )
        result[zone_id] = (coords["x"], coords["y"], coords["w"], coords["h"])

    return result
