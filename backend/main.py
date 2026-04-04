from __future__ import annotations

"""CrowdFlow — FastAPI backend.

Bridges Redis pub/sub to WebSocket clients and provides REST endpoints
for the fan app and demo control.

Run:  uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.websocket_handler import handler as ws_handler
from config import (
    CONCESSION_STALLS,
    CONGESTION_STATES,
    MEDICAL_OVERRIDE_CHANNEL,
    MENU_ITEMS,
    REDIS_URL,
    ZONES,
    classify_congestion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backend")

# ── Redis channels ───────────────────────────────────────────────────────────
CH_ZONE_METRICS      = "broadcast:zone_metrics"
CH_MATCH_CLOCK       = "pos:match_clock"
CH_DASHBOARD_ACTIONS = "dashboard:actions"
CH_SIM_CONTROL       = "sim:control"

# Fan-relevant action types forwarded over the fan WebSocket
FAN_ACTIONS = {
    "FLASH_DEAL",
    "TARGETED_PROMOTION",
    "EMERGENCY_ALERT",
    "EMERGENCY_CLEARED",
    "REROUTE_TRAFFIC",
    "MEDICAL_EMERGENCY",
}

# State severity — used for safety NLP answers
_STATE_SEVERITY: dict[str, int] = {
    "free_flow":      0,
    "moderate":       1,
    "slow_flow":      2,
    "congested":      3,
    "standstill":     4,
    "crush_risk":     5,
    "medical_manual": 6,
}

# ── App state ────────────────────────────────────────────────────────────────
redis_pool: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool
    redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await redis_pool.ping()
        log.info("Connected to Redis")
    except Exception as exc:
        log.error("Redis connection failed: %s", exc)
        raise

    # Start the shared WebSocket broadcast listener as a background task.
    # This single Redis subscriber fans out to all connected dashboard clients.
    listener_task = asyncio.create_task(ws_handler.run_redis_listener(redis_pool))

    yield

    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass
    await redis_pool.aclose()


app = FastAPI(title="CrowdFlow API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket: Dashboard ─────────────────────────────────────────────────────

@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    """Dashboard WebSocket — delegates to the shared DashboardWSHandler.

    The handler maintains a single Redis pub/sub connection that fans out to
    all connected clients, and sends each new client the cached zone state
    immediately on connect.
    """
    await ws_handler.connect(ws)


# ── WebSocket: Fan ───────────────────────────────────────────────────────────

@app.websocket("/ws/fan")
async def ws_fan(ws: WebSocket):
    await ws.accept()
    sub = redis_pool.pubsub()
    try:
        await sub.subscribe(CH_DASHBOARD_ACTIONS)
        log.info("Fan client connected")
        async for msg in sub.listen():
            if msg["type"] != "message":
                continue
            try:
                data = json.loads(msg["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            # Forward approved fan-relevant actions (including MEDICAL_EMERGENCY)
            if data.get("action") in FAN_ACTIONS and data.get("status") == "approved":
                try:
                    await ws.send_json({"channel": "fan:notification", "data": data})
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("Fan WS error: %s", exc)
    finally:
        try:
            await sub.unsubscribe()
            await sub.aclose()
        except Exception:
            pass
        log.info("Fan client disconnected")


# ── REST: Fan Ask ────────────────────────────────────────────────────────────

class FanQuestion(BaseModel):
    question: str


def _is_arabic(text: str) -> bool:
    """Return True if the text contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))


# Arabic keyword sets
_AR_FOOD     = ("أكل", "طعام", "مطعم", "جوعان", "برجر", "مشروب", "أكل سريع", "وجبة")
_AR_BATHROOM = ("حمام", "دورات مياه", "تواليت", "دورة مياه")
_AR_CROWD    = ("زحمة", "كثافة", "حشود", "مزدحم", "فاضي")
_AR_HALFTIME  = ("استراحة", "نص الوقت", "الشوط", "بريك")
_AR_ENDMATCH  = ("خلصت", "انتهت", "نهاية المباراة", "انتهى", "بعد المباراة", "الخروج")
_AR_NAV      = ("وين", "بوابة", "مدخل", "مخرج")
_AR_SAFETY   = ("أمان", "خطر", "طوارئ", "آمن", "سلامة")


@app.post("/api/fan/ask")
async def fan_ask(body: FanQuestion):
    q       = body.question.strip()
    q_lower = q.lower()
    zones_data = await _get_zones_snapshot()
    ar = _is_arabic(q)

    if any(kw in q_lower for kw in ("safe", "safety", "emergency", "danger", "dangerous", "is it safe", "secure")) or \
       any(kw in q for kw in _AR_SAFETY):
        return _answer_safety(zones_data, ar)
    if any(kw in q_lower for kw in ("food", "eat", "hungry", "burger", "drink", "beer", "snack")) or \
       any(kw in q for kw in _AR_FOOD):
        return _answer_food(zones_data, ar)
    if any(kw in q_lower for kw in ("bathroom", "restroom", "toilet", "wc")) or \
       any(kw in q for kw in _AR_BATHROOM):
        return _answer_bathroom(zones_data, ar)
    if any(kw in q_lower for kw in ("crowd", "busy", "packed", "empty", "quiet", "congestion")) or \
       any(kw in q for kw in _AR_CROWD):
        return _answer_crowd(zones_data, ar)
    if any(kw in q_lower for kw in ("end of match", "after match", "leaving", "exit", "full time", "final whistle")) or \
       any(kw in q for kw in _AR_ENDMATCH):
        return _answer_endmatch(zones_data, ar)
    if any(kw in q_lower for kw in ("halftime", "half time", "break", "interval", "half-time")) or \
       any(kw in q for kw in _AR_HALFTIME):
        return _answer_halftime(zones_data, ar)
    if any(kw in q_lower for kw in ("where", "gate", "entrance", "exit")) or \
       any(kw in q for kw in _AR_NAV):
        return _answer_navigation(zones_data, ar)

    if ar:
        return {
            "answer": "أقدر أساعدك بخصوص الأمان، الأكل، دورات المياه، الزحمة، أو الاستراحة. جرّب تسأل عن أحد هذي المواضيع!",
            "type":   "general",
        }
    return {
        "answer": "I can help with safety, food, bathrooms, crowd levels, halftime info, and navigation. Try asking about one of those!",
        "type":   "general",
    }


# ── Answer helpers ───────────────────────────────────────────────────────────

_CONGESTION_AR = {
    "free_flow": "تدفق حر", "moderate": "متوسطة", "slow_flow": "بطيء",
    "congested": "مزدحم",   "standstill": "شبه توقف", "crush_risk": "خطر تزاحم",
    "medical_manual": "طوارئ طبية",
    # legacy fallbacks
    "low": "منخفضة", "high": "عالية", "critical": "حرجة",
}


def _answer_safety(zones: list[dict], ar: bool = False) -> dict:
    """Check for standstill or above; advise accordingly."""
    at_risk = [
        z for z in zones
        if _STATE_SEVERITY.get(
            z.get("congestion_level") or z.get("congestion", "free_flow"), 0
        ) >= _STATE_SEVERITY["standstill"]
    ]
    safe_zones = [
        z for z in zones
        if _STATE_SEVERITY.get(
            z.get("congestion_level") or z.get("congestion", "free_flow"), 0
        ) <= _STATE_SEVERITY["moderate"]
    ]

    if at_risk:
        risk_labels = ", ".join(z["label"] for z in at_risk)
        safe_labels = (
            ", ".join(z["label"] for z in safe_zones[:2])
            if safe_zones else ("outer areas" if not ar else "المناطق الخارجية")
        )
        if ar:
            return {
                "answer": (
                    f"⚠️ تحذير: المناطق التالية بها ازدحام شديد: {risk_labels}. "
                    f"يرجى التوجه إلى {safe_labels} بدلاً من ذلك للسلامة."
                ),
                "type":     "safety",
                "at_risk":  [z["zone_id"] for z in at_risk],
            }
        return {
            "answer": (
                f"⚠️ Caution: {risk_labels} {'are' if len(at_risk) > 1 else 'is'} "
                f"at dangerously high density. Please move to {safe_labels} for your safety."
            ),
            "type":     "safety",
            "at_risk":  [z["zone_id"] for z in at_risk],
        }

    # All zones moderate or below → reassuring
    if ar:
        return {
            "answer": "✅ الملعب آمن الآن. مستويات الحشود طبيعية في جميع المناطق. استمتع بالمباراة!",
            "type":   "safety",
        }
    return {
        "answer": "✅ The stadium is safe right now. Crowd levels are normal across all zones. Enjoy the match!",
        "type":   "safety",
    }


def _answer_food(zones: list[dict], ar: bool = False) -> dict:
    stall_zones = {}
    for sid, cfg in CONCESSION_STALLS.items():
        zid = cfg["zone"]
        for z in zones:
            if z.get("zone_id") == zid:
                stall_zones[sid] = {**cfg, "congestion": z.get("congestion", "free_flow"), "density": z.get("density", 0)}
                break

    best = min(stall_zones.items(), key=lambda x: x[1]["density"])
    stall_id, info = best

    if ar:
        cong = _CONGESTION_AR.get(info["congestion"], info["congestion"])
        return {
            "answer": (
                f"توجه إلى {info['name']} — الأقل ازدحاماً الآن "
                f"(كثافة {cong}). "
                f"الأسعار تبدأ من ${min(m['price'] for m in MENU_ITEMS):.2f}."
            ),
            "type":  "food",
            "stall": stall_id,
        }

    items_str = ", ".join(m["name"] for m in MENU_ITEMS[:4])
    return {
        "answer": (
            f"Head to {info['name']} — it's the least busy right now "
            f"({info['congestion']} congestion). Menu highlights: {items_str}. "
            f"Prices start at ${min(m['price'] for m in MENU_ITEMS):.2f}."
        ),
        "type":  "food",
        "stall": stall_id,
    }


_ZONE_AR = {
    "North Concourse":  "الردهة الشمالية",
    "South Concourse":  "الردهة الجنوبية",
    "North Stand":      "المدرج الشمالي",
    "South Stand":      "المدرج الجنوبي",
    "East Stand":       "المدرج الشرقي",
    "West Stand":       "المدرج الغربي",
    "Main Entrance":    "المدخل الرئيسي",
    "North Gate":       "البوابة الشمالية",
    "South Gate":       "البوابة الجنوبية",
    "Pitch View":       "منطقة الملعب",
    "Activation Zone 1": "منطقة الفعاليات 1",
    "Activation Zone 2": "منطقة الفعاليات 2",
    "North Parking":    "موقف الشمال",
}


def _answer_bathroom(zones: list[dict], ar: bool = False) -> dict:
    concourses = [z for z in zones if "concourse" in z.get("zone_id", "")]
    if concourses:
        best = min(concourses, key=lambda z: z.get("density", 0))
        if ar:
            label = _ZONE_AR.get(best["label"], best["label"])
            cong  = _CONGESTION_AR.get(best.get("congestion", "free_flow"), best.get("congestion", "free_flow"))
            return {
                "answer": (
                    f"أقرب دورة مياه في {label} "
                    f"وهي الأقل ازدحاماً الآن (كثافة {cong}). "
                    f"توجه إليها لتجنب الطوابير."
                ),
                "type": "bathroom",
            }
        return {
            "answer": (
                f"Restrooms are located on each concourse. The {best['label']} "
                f"is least busy right now ({best.get('congestion', 'free_flow')} congestion). "
                f"Head there for shorter queues."
            ),
            "type": "bathroom",
        }
    if ar:
        return {"answer": "دورات المياه متوفرة في جميع الردهات.", "type": "bathroom"}
    return {"answer": "Restrooms are on every concourse level.", "type": "bathroom"}


def _answer_crowd(zones: list[dict], ar: bool = False) -> dict:
    lines = []
    total = sum(z.get("occupancy", 0) for z in zones)
    for z in zones:
        state_key = z.get("congestion_level") or z.get("congestion", "free_flow")
        icon = CONGESTION_STATES.get(state_key, {}).get("icon", "")
        if ar:
            label = _ZONE_AR.get(z["label"], z["label"])
            cong  = _CONGESTION_AR.get(state_key, state_key)
            lines.append(f"  {icon} {label}: {cong} ({z.get('occupancy', 0)}/{z.get('capacity', 0)})")
        else:
            lines.append(f"  {icon} {z['label']}: {state_key} ({z.get('occupancy', 0)}/{z.get('capacity', 0)})")
    overview = "\n".join(lines)
    if ar:
        return {"answer": f"مستويات الحشود الحالية ({total:,} إجمالي):\n{overview}", "type": "crowd"}
    return {"answer": f"Current crowd levels ({total:,} total):\n{overview}", "type": "crowd"}


def _answer_halftime(zones: list[dict], ar: bool = False) -> dict:
    concourses = [z for z in zones if "concourse" in z.get("zone_id", "")]
    if concourses:
        busy  = max(concourses, key=lambda z: z.get("density", 0))
        # Find quietest zone — prefer a different zone than busiest
        quiet_candidates = [z for z in concourses if z["zone_id"] != busy["zone_id"]]
        if not quiet_candidates:
            # Only one concourse — look at all zones for a quieter alternative
            quiet_candidates = [z for z in zones if z["zone_id"] != busy["zone_id"]]
        quiet = min(quiet_candidates, key=lambda z: z.get("density", 0)) if quiet_candidates else busy

        if ar:
            busy_label  = _ZONE_AR.get(busy["label"],  busy["label"])
            quiet_label = _ZONE_AR.get(quiet["label"], quiet["label"])
            return {
                "answer": (
                    f"وقت الاستراحة تزدحم الردهات. حالياً {busy_label} "
                    f"هي الأكثر ازدحاماً. نصيحة: توجه إلى {quiet_label} "
                    f"لطوابير أقصر للأكل ودورات المياه."
                ),
                "type": "halftime",
            }
        return {
            "answer": (
                f"At halftime the concourses get busy. Right now the {busy['label']} "
                f"is the busiest. Pro tip: head to the {quiet['label']} for shorter "
                f"food and restroom queues."
            ),
            "type": "halftime",
        }
    if ar:
        return {"answer": "وقت الاستراحة تزدحم الردهات. خطط لزيارتك مبكراً!", "type": "halftime"}
    return {"answer": "At halftime, concourses get crowded. Plan your trip early!", "type": "halftime"}


def _answer_endmatch(zones: list[dict], ar: bool = False) -> dict:
    """Advice for fans leaving after the final whistle."""
    # Find the least crowded interior zone to shelter in
    interior = [z for z in zones if "stand" in z.get("zone_id", "") or "concourse" in z.get("zone_id", "")]
    quiet = min(interior, key=lambda z: z.get("density", 0)) if interior else None

    if ar:
        quiet_label = _ZONE_AR.get(quiet["label"], quiet["label"]) if quiet else "إحدى المدرجات"
        return {
            "answer": (
                "⚠️ المباراة انتهت والبوابات مزدحمة جداً الآن. "
                f"نصيحة: انتظر 10-15 دقيقة في {quiet_label} حتى تهدأ الزحمة، "
                "ثم اتجه للبوابة الجنوبية أو مواقف الشمال بهدوء."
            ),
            "type": "endmatch",
        }
    quiet_label = quiet["label"] if quiet else "the stands"
    return {
        "answer": (
            "⚠️ The match has ended and the gates are very busy. "
            f"Tip: wait 10–15 minutes in {quiet_label} until the rush eases, "
            "then head to the South Gate or North Parking calmly."
        ),
        "type": "endmatch",
    }


def _answer_navigation(zones: list[dict], ar: bool = False) -> dict:
    # Look for gate or entrance zones
    gate_zone = next(
        (z for z in zones if "gate" in z.get("zone_id", "") or "entrance" in z.get("zone_id", "")),
        None,
    )
    if gate_zone:
        state_key = gate_zone.get("congestion_level") or gate_zone.get("congestion", "free_flow")
        if ar:
            cong = _CONGESTION_AR.get(state_key, state_key)
            return {
                "answer": (
                    f"البوابة/المدخل حالياً بكثافة {cong} "
                    f"({gate_zone.get('occupancy', 0)} شخص). "
                    f"اتبع لوحات الإرشاد في الملعب إلى أقرب بوابة."
                ),
                "type": "navigation",
            }
        return {
            "answer": (
                f"The {gate_zone['label']} currently has {state_key} congestion "
                f"({gate_zone.get('occupancy', 0)} people). "
                f"Follow stadium signage to your nearest gate."
            ),
            "type": "navigation",
        }
    if ar:
        return {"answer": "اتبع لوحات الإرشاد في الملعب إلى أقرب بوابة.", "type": "navigation"}
    return {"answer": "Follow stadium signage to your nearest gate.", "type": "navigation"}


# ── Zone snapshot helper ─────────────────────────────────────────────────────

async def _get_zones_snapshot() -> list[dict]:
    """Get latest zone data from Redis, fall back to config defaults."""
    zones: list[dict] = []
    try:
        # Read individual per-zone keys (written by both camera_streamer and fallback_simulator)
        for zid in ZONES:
            raw = await redis_pool.get(f"zone:{zid}:latest")
            if raw:
                try:
                    zones.append(json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception as exc:
        log.error("Failed to read zone keys: %s", exc)

    if not zones:
        # Fallback: build minimal zone dicts from config
        zones = [
            {
                "zone_id":    zid,
                "label":      cfg["label"],
                "occupancy":  0,
                "capacity":   cfg["capacity"],
                "density":    0.0,
                "congestion": "free_flow",
            }
            for zid, cfg in ZONES.items()
        ]

    # Enrich with congestion_color / congestion_icon if missing
    for z in zones:
        if "congestion_color" not in z or "congestion_icon" not in z:
            key   = z.get("congestion_level") or z.get("congestion", "free_flow")
            state = CONGESTION_STATES.get(key, {})
            z.setdefault("congestion_color", state.get("color", "#22c55e"))
            z.setdefault("congestion_icon",  state.get("icon",  "🟢"))

    return zones


# ── REST: Demo control ───────────────────────────────────────────────────────

async def _publish_control(payload: dict) -> dict:
    try:
        await redis_pool.publish(CH_SIM_CONTROL, json.dumps(payload))
        return {"status": "ok", "action": payload.get("action")}
    except Exception as exc:
        log.error("Control publish failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


@app.post("/api/demo/trigger-halftime")
async def demo_halftime():
    return await _publish_control({"action": "halftime_jump"})


@app.post("/api/demo/trigger-emergency")
async def demo_emergency(zone_id: str = "zone_north_gate", density: float = 6.2):
    return await _publish_control({
        "action":  "emergency",
        "zone_id": zone_id,
        "density": density,
    })


@app.post("/api/demo/trigger-endmatch")
async def demo_endmatch():
    """Set demo:mode to endmatch — fallback simulator will spike exit zones."""
    try:
        await redis_pool.set("demo:mode", "endmatch")
        return {"status": "ok", "action": "endmatch"}
    except Exception as exc:
        log.error("Endmatch trigger failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


@app.post("/api/demo/reset")
async def demo_reset():
    # Clear demo mode
    try:
        await redis_pool.delete("demo:mode")
    except Exception as exc:
        log.error("Demo mode clear failed: %s", exc)
    # Clear medical override state
    try:
        await redis_pool.publish(
            MEDICAL_OVERRIDE_CHANNEL,
            json.dumps({"action": "clear"}),
        )
    except Exception as exc:
        log.error("Medical clear publish failed: %s", exc)
    return await _publish_control({"action": "reset"})


class MedicalRequest(BaseModel):
    zone_id: str
    reason:  str = "Medical emergency — manual override"


@app.post("/api/demo/trigger-medical")
async def demo_trigger_medical(body: MedicalRequest):
    """Publish a medical emergency event to the safety agent."""
    try:
        await redis_pool.publish(
            MEDICAL_OVERRIDE_CHANNEL,
            json.dumps({
                "zone_id":   body.zone_id,
                "reason":    body.reason,
                "timestamp": time.time(),
            }),
        )
        return {"status": "ok", "zone_id": body.zone_id, "reason": body.reason}
    except Exception as exc:
        log.error("Medical trigger failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ── REST: Zones & Health ─────────────────────────────────────────────────────

@app.get("/api/zones")
async def get_zones():
    zones = await _get_zones_snapshot()
    return {"zones": zones, "timestamp": time.time()}


@app.get("/api/health")
async def health():
    redis_ok = False
    try:
        await redis_pool.ping()
        redis_ok = True
    except Exception:
        pass
    return {
        "status":    "healthy" if redis_ok else "degraded",
        "redis":     redis_ok,
        "timestamp": time.time(),
    }


# ── Static file serving for frontends ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


app.mount("/data", StaticFiles(directory=str(PROJECT_ROOT / "data")), name="data")


@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(PROJECT_ROOT / "dashboard" / "index.html")


@app.get("/fan")
async def serve_fan():
    return FileResponse(PROJECT_ROOT / "fan-webapp" / "index.html")
