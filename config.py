"""CrowdFlow — centralised configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# ── Stadium zone definitions ─────────────────────────────────────────────────
# Each zone: label, capacity (persons), area_m2, camera_source,
#            bounding box (x, y, w, h) in pixels, zone_type
ZONES = {
    # ── Interior zones (7 original) ─────────────────────────────────────────
    "zone_north_gate": {
        "label": "North Gate",
        "capacity": 8000,
        "area_m2": 7200.0,       # 120 × 30 px × 2 m²/px
        "camera_source": None,
        "x": 0, "y": 0, "w": 120, "h": 30,
        "zone_type": "interior",
    },
    "zone_south_gate": {
        "label": "South Gate",
        "capacity": 8000,
        "area_m2": 7200.0,       # 120 × 30 px × 2 m²/px
        "camera_source": None,
        "x": 0, "y": 90, "w": 120, "h": 30,
        "zone_type": "interior",
    },
    "zone_east_stand": {
        "label": "East Stand",
        "capacity": 5000,
        "area_m2": 7200.0,       # 30 × 120 px × 2 m²/px
        "camera_source": None,
        "x": 120, "y": 0, "w": 30, "h": 120,
        "zone_type": "interior",
    },
    "zone_west_stand": {
        "label": "West Stand",
        "capacity": 5000,
        "area_m2": 7200.0,       # 30 × 120 px × 2 m²/px
        "camera_source": None,
        "x": -30, "y": 0, "w": 30, "h": 120,
        "zone_type": "interior",
    },
    "zone_north_concourse": {
        "label": "North Concourse",
        "capacity": 3000,
        "area_m2": 2400.0,       # 120 × 10 px × 2 m²/px
        "camera_source": None,
        "x": 0, "y": -10, "w": 120, "h": 10,
        "zone_type": "interior",
    },
    "zone_south_concourse": {
        "label": "South Concourse",
        "capacity": 3000,
        "area_m2": 2400.0,       # 120 × 10 px × 2 m²/px
        "camera_source": None,
        "x": 0, "y": 120, "w": 120, "h": 10,
        "zone_type": "interior",
    },
    "zone_pitch_view": {
        "label": "Pitch View",
        "capacity": 2000,
        "area_m2": 600.0,        # 30 × 10 px × 2 m²/px
        "camera_source": None,
        "x": 45, "y": -20, "w": 30, "h": 10,
        "zone_type": "interior",
    },
    # ── Exterior zones (3 new) ───────────────────────────────────────────────
    "zone_activation_1": {
        "label": "Activation Zone 1",
        "capacity": 2000,
        "area_m2": 5000.0,
        "camera_source": None,
        "x": 520, "y": 50, "w": 160, "h": 120,
        "zone_type": "exterior",
    },
    "zone_activation_2": {
        "label": "Activation Zone 2",
        "capacity": 1500,
        "area_m2": 3500.0,
        "camera_source": None,
        "x": 520, "y": 200, "w": 160, "h": 100,
        "zone_type": "exterior",
    },
    "zone_north_parking": {
        "label": "North Parking",
        "capacity": 3000,
        "area_m2": 8000.0,
        "camera_source": None,
        "x": 100, "y": -80, "w": 300, "h": 70,
        "zone_type": "exterior",
    },
}

# ── Concession stalls ────────────────────────────────────────────────────────
CONCESSION_STALLS = {
    "stall_A": {"name": "North Bites",  "zone": "zone_north_concourse", "x": 20,  "y": -5},
    "stall_B": {"name": "South Grill",  "zone": "zone_south_concourse", "x": 60,  "y": 125},
    "stall_C": {"name": "East Drinks",  "zone": "zone_east_stand",      "x": 135, "y": 60},
    "stall_D": {"name": "West Snacks",  "zone": "zone_west_stand",      "x": -15, "y": 60},
}

# ── Menu items & prices (USD) ────────────────────────────────────────────────
MENU_ITEMS = [
    {"id": "burger",  "name": "Classic Burger", "price": 9.50},
    {"id": "hotdog",  "name": "Hot Dog",         "price": 6.00},
    {"id": "nachos",  "name": "Loaded Nachos",   "price": 8.00},
    {"id": "soda",    "name": "Soft Drink",       "price": 4.50},
    {"id": "beer",    "name": "Draft Beer",       "price": 10.00},
    {"id": "water",   "name": "Water Bottle",     "price": 3.00},
    {"id": "fries",   "name": "Fries",            "price": 5.50},
    {"id": "pretzel", "name": "Soft Pretzel",     "price": 7.00},
]

# ── Congestion state hierarchy ───────────────────────────────────────────────
# Each state: name, color (hex), icon (emoji), density_range (persons/m²), description
CONGESTION_STATES = {
    "free_flow": {
        "name": "Free Flow",
        "color": "#22c55e",
        "icon": "🟢",
        "density_range": (0.0, 1.0),
        "description": "Free movement, no delays",
    },
    "moderate": {
        "name": "Moderate",
        "color": "#84cc16",
        "icon": "🟡",
        "density_range": (1.0, 2.0),
        "description": "Medium congestion, slight slowdown",
    },
    "slow_flow": {
        "name": "Slow Flow",
        "color": "#eab308",
        "icon": "🟠",
        "density_range": (2.0, 3.0),
        "description": "Slow movement, noticeable delays",
    },
    "congested": {
        "name": "Congested",
        "color": "#f97316",
        "icon": "🔶",
        "density_range": (3.0, 4.0),
        "description": "High congestion, long wait times",
    },
    "standstill": {
        "name": "Standstill",
        "color": "#ef4444",
        "icon": "🔴",
        "density_range": (4.0, 5.5),
        "description": "Near standstill, crowd crush risk",
    },
    "crush_risk": {
        "name": "Crush Risk",
        "color": "#dc2626",
        "icon": "🚨",
        "density_range": (5.5, 7.0),
        "description": "Critical density, immediate danger",
    },
    "medical_manual": {
        "name": "Medical Emergency",
        "color": "#7c3aed",
        "icon": "🏥",
        "density_range": None,
        "description": "Manual medical emergency override",
    },
}


def classify_congestion(occupancy_pct: float) -> str:
    """Return the CONGESTION_STATES key for the given occupancy percentage."""
    if occupancy_pct < 40:
        return "free_flow"
    if occupancy_pct < 60:
        return "moderate"
    if occupancy_pct < 80:
        return "slow_flow"
    if occupancy_pct < 95:
        return "congested"
    if occupancy_pct < 105:
        return "standstill"
    return "crush_risk"


def get_state_config(state_key: str) -> dict:
    """Return the full config dict for a congestion state key."""
    return CONGESTION_STATES[state_key]


# ── Camera feeds ─────────────────────────────────────────────────────────────
CAMERA_FEEDS = {
    "cam_parking": {
        "zone_id":    "zone_north_parking",
        "video_file": "data/videos/north_gate.mp4",
        "fps":        5,
    },
}

# ── Agent threshold constants ─────────────────────────────────────────────────
DENSITY_SWITCH_THRESHOLD = 2.0       # persons/m² — crowd-management mode switch

SLOW_FLOW_DENSITY   = 2.0            # persons/m²
STANDSTILL_DENSITY  = 4.0            # persons/m²
CRUSH_RISK_DENSITY  = 5.5            # persons/m²

# Legacy aliases kept for backward compat with safety_agent.py and orchestrator.py
CONGESTION_MODERATE = 2.0            # persons/m²
CONGESTION_HIGH     = 3.5            # persons/m²

SAFETY_MAX_DENSITY  = 6.0            # persons/m² — hard evacuation trigger
SAFETY_MAX_FLOW_RATE = 80            # persons/minute through a gate
SAFETY_EMERGENCY_COUNT = 800         # persons — aggregate emergency threshold

# Growth-rate alerts (% increase per tick)
GROWTH_RATE_WARNING  = 15.0
GROWTH_RATE_CRITICAL = 30.0

# ── Redis pub/sub channel names ───────────────────────────────────────────────
CHANNEL_ZONE_UPDATE       = "crowdflow:zone_update"
CHANNEL_AGENT_ACTION      = "crowdflow:agent_action"
CHANNEL_ALERT             = "crowdflow:alert"
CHANNEL_CONCESSION_ORDER  = "crowdflow:concession_order"
CHANNEL_FAN_NOTIFICATION  = "crowdflow:fan_notification"
CH_CAMERA_INFERENCE       = "camera:inference_results"
MEDICAL_OVERRIDE_CHANNEL  = "manual:medical_emergency"
