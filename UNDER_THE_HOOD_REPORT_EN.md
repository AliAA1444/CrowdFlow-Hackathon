# CrowdFlow — Under-the-Hood Technical Report

> **Author:** Senior Software Architect & Code Auditor
> **Mode:** Read-Only audit. **No source files were modified during this audit.**
> **Scope:** Exact identifiers, thresholds, and code paths as they exist in the repository today — not roadmap or aspirational claims.
> **Audience:** Interview prep — every claim is anchored to a file path and (where useful) a line number.

---

## 0. Architecture Map (Context)

```
+---------------------------------------------------------------+
|  9 video feeds  (feeds/*.mp4 | *.mov)                         |
+---------------------------------------------------------------+
              |
              v
+---------------------------------------------------------------+
| vision/pipeline.py  ::  VisionPipeline                        |
|   - YOLOv8n  (ultralytics.YOLO("yolov8n.pt"))                 |
|   - MOG2     (cv2.createBackgroundSubtractorMOG2)             |
|   - Farneback flow (cv2.calcOpticalFlowFarneback)             |
|   - ByteTrack (gate zones, via model.track(persist=True))     |
|   - asyncio.Semaphore(3)  +  PROCESS_EVERY_N_FRAMES = 3       |
+---------------------------------------------------------------+
              |
              v   ZonePublisher  (core/redis_publisher.py)
+---------------------------------------------------------------+
| Redis  ::  zone:{id}:latest        (SET, TTL = 30 s)          |
|         ::  PUBLISH "zone_updates"                            |
+---------------------------------------------------------------+
              |
              v
+---------------------------------------------------------------+
| agents/agent_runner.py  ::  AgentRunner                       |
|   - SafetyAgent / CrowdFlowAgent / ConcessionAgent.evaluate() |
|   - CooldownManager (cooldown_manager.py)                     |
|   - PUBLISH "agent_actions"                                   |
+---------------------------------------------------------------+
              |
              v
+---------------------------------------------------------------+
| backend/main.py  +  api/websocket_handler.py                  |
|   - FastAPI + Uvicorn (port 8000)                             |
|   - DashboardWSHandler  (single Redis sub → fan-out to N WS)  |
|   - /ws/dashboard, /ws/fan, /api/fan/ask, /api/demo/*         |
+---------------------------------------------------------------+
              |
              v
+---------------------------------------------------------------+
| dashboard/index.html  +  fan-webapp/index.html                |
+---------------------------------------------------------------+
```

Two boot scripts: `run_demo.py` (the production-grade unified launcher) and `run_all.py` (legacy launcher that also starts `Orchestrator` and `CameraInferenceEngine`).

---

## 1. Computer Vision Engine & Exact Metrics

### 1.1 Models in Use

**Primary detector — `vision/pipeline.py:9-10, 49, 70`:**

```python
from ultralytics import YOLO
...
self.model = YOLO(yolo_model_path)        # default: "yolov8n.pt"
```

The system is **hybrid** — it does **not** rely on YOLO alone. Four CV engines run in parallel for each zone:

| Engine | Defining file | Role |
|---|---|---|
| **YOLOv8-nano** (object detector) | `vision/pipeline.py` | Person count for general & gate zones |
| **MOG2** background subtractor | `vision/density_estimator.py` | Density estimate when YOLO saturates (heavy occlusion) |
| **Farneback** dense optical flow | `vision/flow_analyzer.py` | Flow direction (degrees) and magnitude (px/frame) |
| **YOLO + ByteTrack** | `vision/gate_counter.py` | Tripwire IN/OUT counting at gates |

YOLOv8n is a single-shot detector with a CSP-Darknet backbone and an anchor-free head; only `class=0` (`person` in COCO) is requested.

### 1.2 Exact Numerical Thresholds (extracted verbatim)

#### YOLO call sites

| Call site | File:Line | `conf` | `imgsz` | `classes` | Notes |
|---|---|---|---|---|---|
| General zone detection | `vision/pipeline.py:168` | **0.15** | **1280** | `[0]` | Low conf catches partial occlusions; large `imgsz` recovers distant/small persons |
| Gate instantaneous count | `vision/pipeline.py:143` | **0.25** | (default) | `[0]` | Higher conf — gates need precision more than recall |
| Gate tripwire tracker | `vision/gate_counter.py:55-62` | **0.15** | **1280** | `[0]` | `persist=True` to maintain ByteTrack state across frames |
| `tools/cv_playground.py` | line 99 (default), line 105 | **0.15** | **1280** | `[0]` | CLI-overridable |

There is **no explicit `max_det`/`max_detections` argument anywhere in the code**, so the practical detection cap per frame is Ultralytics' default (`max_det = 300`). The system relies on the YOLO→MOG2 mode switch (described below) to handle scenes that would otherwise saturate that limit.

#### MOG2 background subtractor — `vision/density_estimator.py:40-42`

```python
self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)
```

| Parameter | Value | File:Line | Meaning |
|---|---|---|---|
| `history` | **500** | density_estimator.py:41 | Frames used to build the background model |
| `varThreshold` | **50** | density_estimator.py:41 | Mahalanobis distance threshold for fg/bg classification |
| `detectShadows` | **False** | density_estimator.py:42 | Disabled (cheaper, simpler mask) |
| `_WARMUP_FRAMES` | **50** | density_estimator.py:10 | First 50 calls return `-1` ("not ready") |
| `learning_rate` | **0.005** | density_estimator.py:25 | MOG2 background update rate |
| `min_area` | **100** px | density_estimator.py:28 | Lowered from 500 to keep distant person blobs |
| `calibration_factor` (default) | **0.005** | density_estimator.py:24 | Persons-per-foreground-pixel; `config/zones.json` uses **0.0025** for every general zone |
| Morphological kernel | `(3, 3) ELLIPSE` | density_estimator.py:68 | Reduced from `(5,5)` to preserve sub-5px blobs |

Final count formula:

```python
fg_pixels = cv2.countNonZero(cleaned_mask)
count     = int(fg_pixels * self.calibration_factor)
```

#### YOLO ↔ MOG2 mode switch — `vision/pipeline.py:55, 181-198`

```python
switch_threshold: int = 100
...
if yolo_count >= self.switch_threshold:
    dense_count = dense_est.estimate(frame, roi)
    if dense_count == -1:                 # MOG2 still warming up
        raw_count = yolo_count
        source    = "cv_yolo"
    else:
        raw_count = dense_count
        source    = "cv_dense" (+ "cal@<factor>:<conf>%" suffix if calibrator confident)
else:
    raw_count = yolo_count
    source    = "cv_yolo"
```

> **Audit nit:** the same logical switch in `tools/cv_playground.py:299` is hardcoded at **40**, not 100. This divergence will surface if you debug a zone with the playground and then expect identical behaviour in production.

#### Cross-calibrator (online MOG2 tuning) — `vision/cross_calibrator.py:17-26`

```python
learning_rate    = 0.05
min_yolo_count   = 10
max_yolo_count   = 120     # raised from 45 to learn from medium-density scenes
min_fg_pixels    = 1000
```

EMA update on `calibration_factor`:

```python
observed_factor = yolo_count / fg_pixels
self._current_factor = (
    self.learning_rate * observed_factor
    + (1.0 - self.learning_rate) * self._current_factor
)
```

Outlier rejection (cross_calibrator.py:75): any sample whose ratio to the current factor is `> 100x` or `< 0.01x` is dropped. Confidence saturates at `1.0` after `samples / 20` (i.e. 20 valid frames).

#### Farneback dense optical flow — `vision/flow_analyzer.py:44-55`

```python
flow = cv2.calcOpticalFlowFarneback(
    self._prev_gray, gray_frame, None,
    pyr_scale = 0.5,
    levels    = 3,
    winsize   = 15,
    iterations= 3,
    poly_n    = 5,
    poly_sigma= 1.2,
    flags     = 0,
)
```

| Constant | Value | File:Line |
|---|---|---|
| `STAGNATION_THRESHOLD` | **0.5** px/frame | flow_analyzer.py:12 |

Direction is computed with `(np.degrees(np.arctan2(mean_dy, mean_dx)) + 360) % 360`. `is_stagnant = magnitude < STAGNATION_THRESHOLD`.

#### Gate tripwire counter — `vision/gate_counter.py:13-15`

```python
_DIRECTION_LABELS = {"up", "down", "left", "right"}
_HISTORY_MAX     = 30      # last N centroid positions retained per track
_PRUNE_INTERVAL  = 1000    # every 1000 frames, drop counted IDs that no longer track
```

Crossing detection uses 2D segment intersection (`_segments_intersect`, line 142-160) and the cross product of the tripwire vector with the movement vector (`_crossing_direction`, line 162-196) to disambiguate IN vs OUT.

#### Smoothing — `core/smoothing.py` :: `EMACounter`

| Parameter | Value | File:Line |
|---|---|---|
| `alpha` | **0.3** | smoothing.py:29 |
| `spike_threshold` | **2.0** (= 200% relative change) | smoothing.py:33 |
| `N_OVERRIDE` | **5** consecutive rejections → force-accept raw | smoothing.py:25 |
| Zero-guard | `raw==0 ∧ ema>10` → reject (does **not** count toward `N_OVERRIDE`) | smoothing.py:58 |

### 1.3 Frame Skipping — math, logic, and CPU protection

**File:** `vision/pipeline.py:25, 248-294`.

```python
PROCESS_EVERY_N_FRAMES = 3        # module-level constant

async def run_feed(self, zone_id, video_source, zone_type):
    cap = cv2.VideoCapture(video_source)
    frame_index = 0
    while True:
        ret, frame = await asyncio.to_thread(cap.read)        # (1)
        ...
        frame_index += 1
        if frame_index % PROCESS_EVERY_N_FRAMES != 0:         # (2)
            continue
        ...
        async with self._inference_semaphore:                 # (3)
            await self.process_frame(zone_id, frame, zone_type)
```

**Step-by-step:**

1. **`cap.read()` runs every frame**, wrapped in `asyncio.to_thread()` so the OpenCV I/O does **not** block the event loop. Skipping `cap.read()` would cause clock drift — the video would appear to fast-forward at 3× speed.
2. **`frame_index % PROCESS_EVERY_N_FRAMES != 0` continues without invoking heavy CV.** Only frames where `frame_index ≡ 0 (mod 3)` (i.e. frames 3, 6, 9, …) trigger the YOLO/MOG2/ByteTrack/Farneback chain. Inference cost drops to `1/3 ≈ 33%` of full-rate; the in-code comment claims **"~70% CPU reduction"**.
3. **Concurrency cap: `asyncio.Semaphore(3)`** (`vision/pipeline.py:67`). Even with 9 concurrent feeds, **at most 3 zones execute heavy CV simultaneously**. A 4th zone awaits the semaphore — preventing CPU saturation on an 8-core MacBook Air (per the in-code comment).

Per-zone FPS is computed from **processed frames only** (line 287-290):

```python
now      = time.perf_counter()
elapsed  = now - prev_time
self._zone_fps[zone_id] = 1.0 / elapsed if elapsed > 0 else 0.0
```

`config/zones.json` mirrors these knobs in its `demo_settings` block:

```json
"demo_settings": {
    "process_every_n_frames": 3,
    "max_concurrent_inference": 3
}
```

---

## 2. The 9 Zones & Isolated Video Testing

### 2.1 The exact 9 zones (production source: `config/zones.json`)

| # | `zone_id` | `type` | `area_sqm` | `sector_total_sqm` | `capacity` | `video_source` | Calibration / Tripwire |
|---|---|---|---|---|---|---|---|
| 1 | `gate_main` | gate | 80 | 80 | 400 | `feeds/real_gate_main.mp4` | tripwire `[640,0]→[640,720]`, IN=`right` |
| 2 | `gate_vip` | gate | 60 | 60 | 300 | `feeds/real_gate_vip.mp4` | tripwire `[0,360]→[1280,360]`, IN=`down` |
| 3 | `gate_south` | gate | 70 | 70 | 350 | `feeds/fan_plaza.mov` | tripwire `[640,0]→[640,720]`, IN=`right` |
| 4 | `concourse_main` | general | 350 | 2 800 | — | `feeds/real_concourse.mp4` | `calibration_factor = 0.0025` |
| 5 | `stands_east` | general | 300 | 3 000 | — | `feeds/real_stands_dense.mp4` | `0.0025` |
| 6 | `stands_west` | general | 300 | 3 000 | — | `feeds/stands_west.mov` | `0.0025` |
| 7 | `stands_north` | general | 280 | 2 800 | — | `feeds/stands_north.mov` | `0.0025` |
| 8 | `food_court` | general | 400 | 1 600 | — | `feeds/food_court.mp4` | `0.0025` |
| 9 | `activation_zone` | general | 500 | 2 000 | — | `feeds/activation_zone.mov` | `0.0025` |

**Coverage-ratio extrapolation** — `vision/pipeline.py:151-156, 203-207`:

```python
sector_sqm     = float(cfg.get("sector_total_sqm", area_sqm))
coverage_ratio = sector_sqm / max(area_sqm, 1.0)
if coverage_ratio > 1.0:
    raw_count = int(raw_count * coverage_ratio)
    source    = f"{source}:x{coverage_ratio:.0f}"
```

The camera covers `area_sqm` of the full sector. The displayed count is scaled by `sector_total_sqm / area_sqm`, but density (count/area) is invariant because numerator and denominator scale together.

> **Note on the legacy `ZONES` dict in `config.py`:** that file declares 10 zones (7 interior + 3 exterior — `zone_north_gate`, `zone_south_gate`, `zone_east_stand`, `zone_west_stand`, `zone_north_concourse`, `zone_south_concourse`, `zone_pitch_view`, `zone_activation_1`, `zone_activation_2`, `zone_north_parking`). It is consumed by the legacy crowd simulator, the legacy `Orchestrator`, parts of `CrowdFlowAgent` (overflow targets), and the FastAPI fallback list. The CV pipeline production path is driven exclusively by the 9 zones in `config/zones.json` above.

### 2.2 Isolated single-feed testing — exact mechanism

**The CLI tool is `tools/cv_playground.py`.** It runs the **same vision components as production** (YOLO, `DenseEstimator`, `CrossCalibrator`, `OpticalFlowAnalyzer`, `GateTripwireCounter`, `EMACounter`) with **zero backend dependencies** — no Redis, no FastAPI, no WebSockets, no agents.

The mapping between zones and video files is **not** hand-coded in the test tool. The tool **reads `config/zones.json` at startup** and pulls every relevant setting (video path, ROI, area, mode, calibration factor, tripwire, IN direction) for the selected `zone_id`.

#### Path A — Auto-load by zone id (preferred)

```bash
python tools/cv_playground.py --zone concourse_main
```

Internal logic (`tools/cv_playground.py:139-183`) — when `--zone` is provided:

```
1. Open config/zones.json
2. Look up zones[args.zone]
3. Override:
     args.video             ← zcfg["video_source"]
     args.mode              ← zcfg["type"]                  # "general" or "gate"
     args.area_sqm          ← zcfg["area_sqm"]
     args.roi               ← json.dumps(zcfg["roi"])
     args.calibration_factor ← zcfg["calibration_factor"]   # general only
     args.tripwire          ← json.dumps([tw["p1"], tw["p2"]])  # gate only
     args.in_direction      ← zcfg["in_direction"]          # gate only
```

This is the **single source of truth**: `config/zones.json` is the only file that ties video files to zone IDs. There are no env vars, no per-zone test scripts, no symlinks.

#### Path B — Manual override (when you want to point at an arbitrary video)

```bash
# General (density) zone
python tools/cv_playground.py feeds/real_concourse.mp4 \
    --mode general \
    --roi '{"x":85,"y":42,"w":1110,"h":636}' \
    --area-sqm 400 \
    --calibration-factor 0.0025

# Gate zone (ByteTrack tripwire)
python tools/cv_playground.py feeds/real_gate_main.mp4 \
    --mode gate \
    --tripwire '[[640,0],[640,720]]' \
    --in-direction right
```

#### CLI flags reference

| Flag | Default | Effect |
|---|---|---|
| `video` (positional) | — | Path to MP4/MOV file (or set via `--zone`) |
| `--zone <id>` | — | Auto-load all settings from `config/zones.json` |
| `--mode {general, gate}` | `general` | Pipeline mode |
| `--roi '{"x":..,"y":..,"w":..,"h":..}'` | full frame | ROI as JSON |
| `--area-sqm <float>` | `400.0` | Zone area in m² |
| `--calibration-factor <float>` | `0.0025` (when `--zone`) | MOG2 calibration |
| `--tripwire '[[x1,y1],[x2,y2]]'` | required for gate | Tripwire JSON |
| `--in-direction {up,down,left,right}` | `down` | Gate IN direction |
| `--conf <float>` | `0.15` | YOLO confidence |
| `--imgsz <int>` | `1280` | YOLO input size |
| `--headless` | off | No GUI; print stats |
| `--export-csv <path>` | — | Per-frame metrics CSV (yolo_count, dense_count, fg_pixels, cal_factor, smoothed, density, flow_dir, flow_mag, …) |
| `--slow` | off | Step-through mode (key press per frame) |
| `--no-yolo-overlay` | off | Hide YOLO bboxes |

Live keys while the OpenCV window is focused: `SPACE` pause/resume, `Q` quit, `S` save current debug frame, `N` next frame (when paused).

#### What `process_frame_general` actually does (`tools/cv_playground.py:277-337`)

```python
1.  results       = yolo(frame, conf=0.15, classes=[0], imgsz=1280)
    yolo_count    = len(results[0].boxes)
2.  dense_count   = dense_estimator.estimate(frame, (x,y,w,h))
    fg_pixels     = dense_estimator.get_last_fg_pixels()
3.  cal_factor    = cross_calibrator.update(yolo_count, fg_pixels)
4.  # Mode switch (NB: hardcoded 40 here vs 100 in production)
    if yolo_count >= 40 and dense_count >= 0:
        chosen_count = dense_count;  mode = "DENSE (MOG2)"
    else:
        chosen_count = yolo_count;   mode = "YOLO"
5.  smoothed      = ema.update(chosen_count)         # alpha=0.3, spike=2.0
6.  flow_results  = flow_analyzer.analyze(frame)
7.  density       = smoothed / area_sqm
```

UI panels rendered (general mode): YOLO overlay (top-left), MOG2 mask + `fg_pixels`/`cal_factor` (top-right), Farneback arrow + `STAGNANT` flag (bottom-left), full stats text panel (bottom-right). For gate mode: gate overlay with tripwire line + IN/OUT counters (left), gate stats panel (right).

---

## 3. AI Agents & Cooldown Logic

### 3.1 SafetyAgent (`agents/safety_agent.py`) — priority 0

#### Live thresholds used by `evaluate(update: ZoneUpdate)` — lines 51-75

| Constant | Value | Triggers | Priority |
|---|---|---|---|
| `SMALL_DENSITY_CRITICAL` | **0.45** p/m² | `CRITICAL: Zone at maximum safe capacity` | `critical` |
| `SMALL_DENSITY_CRITICAL_CLEAR` | **0.37** p/m² | (hysteresis only) | — |
| `SMALL_DENSITY_WARNING` | **0.28** p/m² | `WARNING: Zone approaching capacity limit` | `high` |
| `SMALL_DENSITY_WARNING_CLEAR` | **0.18** p/m² | (hysteresis only) | — |
| `SMALL_FLOW_SURGE` | **4.0** px/frame | `ALERT: Rapid crowd surge detected …` (also requires `density > 0.20`) | `critical` |
| `SMALL_FLOW_SURGE_MIN_DENSITY` | **0.20** p/m² | gating condition above | — |
| `SMALL_DENSITY_OVERLOAD` | **0.25** p/m² | `SYSTEM_OVERLOAD …` (also requires every neighbour `> SMALL_ADJACENT_PRESSURE`) | `critical` |
| `SMALL_ADJACENT_PRESSURE` | **0.15** p/m² | "neighbour under pressure" cutoff | — |
| `SMALL_STAGNATION_MIN_DENSITY` | **0.30** p/m² | `WARNING: Crowd stagnation …` (also requires `flow_magnitude < 0.3`) | `high` |

Verbatim trigger logic (`safety_agent.py:147-241`):

```python
# Density rules
if d > SMALL_DENSITY_CRITICAL:                                  # > 0.45
    actions.append(AgentAction(action="CRITICAL: …", priority="critical", …))
elif d > SMALL_DENSITY_WARNING:                                 # > 0.28
    actions.append(AgentAction(action="WARNING: …", priority="high", …))

# Flow surge
if mag > SMALL_FLOW_SURGE and d > SMALL_FLOW_SURGE_MIN_DENSITY:  # > 4.0 AND > 0.20
    actions.append(AgentAction(action="ALERT: Rapid crowd surge detected — …",
                               priority="critical", …))

# Stagnation
if mag < 0.3 and d > SMALL_STAGNATION_MIN_DENSITY:               # < 0.3 AND > 0.30
    actions.append(AgentAction(action="WARNING: Crowd stagnation detected — …",
                               priority="high", …))

# Cross-zone overload
if d > SMALL_DENSITY_OVERLOAD and self._zone_cache is not None:  # > 0.25
    adjacent = self._zone_cache.get_adjacent_zones(zid)
    if adjacent and all(a.density > SMALL_ADJACENT_PRESSURE for a in adjacent):  # all > 0.15
        actions.append(AgentAction(action="SYSTEM_OVERLOAD: …", priority="critical", …))
```

#### Legacy thresholds (still active for the `broadcast:zone_metrics` channel) — lines 39-43

| Constant | Value | Action emitted |
|---|---|---|
| `DENSITY_CRUSH_RISK` (= `CRUSH_RISK_DENSITY`) | **5.5** p/m² | `EMERGENCY_ALERT` (priority=0) |
| `DENSITY_STANDSTILL` (= `STANDSTILL_DENSITY`) | **4.0** p/m² | `SAFETY_WARNING tier=standstill` (priority=1) |
| `DENSITY_WARNING` (= `CONGESTION_HIGH`) | **3.5** p/m² | `SAFETY_WARNING tier=warning` (priority=1) |
| `COUNT_EMERGENCY` | **800** persons | Alternative trigger for `EMERGENCY_ALERT` |
| `CLEAR_FACTOR` | **0.80** | Hysteresis — clear when `density < 5.5×0.80 ∧ count < 800×0.80` |

Plus, in `config.py:217-219`:

```python
SAFETY_MAX_DENSITY     = 6.0     # p/m² — hard evacuation trigger
SAFETY_MAX_FLOW_RATE   = 80      # persons/minute through a gate
SAFETY_EMERGENCY_COUNT = 800
```

### 3.2 CrowdFlowAgent (`agents/crowd_flow_agent.py`) — priority 2

#### Live thresholds in `evaluate()` — lines 45-51

| Constant | Value | Action |
|---|---|---|
| `SMALL_REROUTE_DENSITY` | **0.25** p/m² | `REROUTE_FANS` (priority `high`) |
| `SMALL_ADJACENT_FREE` | **0.12** p/m² | Adjacent zone density must be below this to be a valid reroute target |
| `SMALL_GATE_NET_IMBALANCE` | **40** persons | `GATE_THROTTLE` when `gate_in - gate_out > 40` |

Verbatim:

```python
# Reroute logic
if d > SMALL_REROUTE_DENSITY and self._zone_cache is not None:    # > 0.25
    for adj in self._zone_cache.get_adjacent_zones(zid):
        if adj.density < SMALL_ADJACENT_FREE:                     # < 0.12
            actions.append(AgentAction(action="REROUTE_FANS", priority="high", …))
            break

# Gate throttle
if zid.startswith("gate_"):
    net = update.gate_in - update.gate_out
    if net > SMALL_GATE_NET_IMBALANCE:                            # > 40
        actions.append(AgentAction(action="GATE_THROTTLE", priority="high", …))
```

#### Legacy growth-trend logic — lines 36-40

| Constant | Value | Meaning |
|---|---|---|
| `ROLLING_WINDOW` | **30** | Last N readings retained per zone |
| `RECENT_WINDOW` | **5** | Most-recent N readings to compare … |
| `OLDER_WINDOW` | **5** | … against the previous N readings |
| `GROWTH_THRESHOLD` | **25.0 %** | Triggers `REROUTE_TRAFFIC` (also needs `density >= SLOW_FLOW_DENSITY = 2.0`) |
| `PREDICTION_MINUTES` | **5.0** | Horizon for linear extrapolation `_predict()` |

Plus, in `config.py:222-223`:

```python
GROWTH_RATE_WARNING  = 15.0   # %
GROWTH_RATE_CRITICAL = 30.0   # %
```

Growth math (`crowd_flow_agent.py:267-283`):

```python
older  = items[-(RECENT_WINDOW + OLDER_WINDOW):-RECENT_WINDOW]
recent = items[-RECENT_WINDOW:]
growth = ((mean(recent) - mean(older)) / mean(older)) * 100.0
```

### 3.3 ConcessionAgent (`agents/concession_agent.py`) — priority 3

#### Live thresholds in `evaluate()` — lines 50-64

| Constant | Value | Action |
|---|---|---|
| `SMALL_FLASH_MAX_DENSITY` | **0.12** p/m² | `FLASH_SALE` (priority `low`) — combined with: |
| `SMALL_FLASH_MIN_FLOW` | **1.8** px/frame | Foot-traffic above noise floor |
| `SMALL_PAUSE_DENSITY` | **0.30** p/m² | `PAUSE_PROMOTIONS` (priority `medium`) |
| `SMALL_GATE_INFLUX` | **25** persons (`gate_in`) | `WELCOME_DEAL` — combined with: |
| `SMALL_GATE_MAX_DENSITY` | **0.18** p/m² | Gate not yet jammed |

Verbatim:

```python
# Flash sale: passing traffic, low dwell
if d < SMALL_FLASH_MAX_DENSITY and mag > SMALL_FLASH_MIN_FLOW:    # < 0.12 AND > 1.8
    actions.append(AgentAction(action="FLASH_SALE", priority="low", …))

# Pause promotions: zone congested
elif d > SMALL_PAUSE_DENSITY:                                      # > 0.30
    actions.append(AgentAction(action="PAUSE_PROMOTIONS", priority="medium", …))

# Welcome deal: arriving fans, gate still manageable
if zid.startswith("gate_") and update.gate_in > SMALL_GATE_INFLUX \
        and d < SMALL_GATE_MAX_DENSITY:                            # > 25 AND < 0.18
    actions.append(AgentAction(action="WELCOME_DEAL", priority="low", …))
```

#### Legacy POS / intent thresholds — lines 40-44

| Constant | Value | Action |
|---|---|---|
| `HIGH_VELOCITY_THRESHOLD` | **40** tx/min | `DYNAMIC_PRICE_UP` (multiplier `1.15`) |
| `VELOCITY_WINDOW` | **60.0** s | Tx velocity sliding window |
| `LOW_TRAFFIC_UTILIZATION` | **0.25** | Zone utilization below this ⇒ candidate for legacy `FLASH_DEAL` |
| `FLASH_DEAL_COOLDOWN` | **120.0** s | Local per-zone cooldown for legacy flash deals |
| `INTENT_THRESHOLD` | **15** intents | Triggers `TARGETED_PROMOTION` |
| `INTENT_WINDOW` | **60.0** s | Intent accumulation window |

Surge de-escalation: `surge_active` flag is reset when `velocity < HIGH_VELOCITY_THRESHOLD * 0.7` (= 28 tx/min).

### 3.4 Consolidated trigger matrix — exact math per category

| Category | Action | Triggering condition | Agent |
|---|---|---|---|
| **Overload / Critical** | `CRITICAL: Zone at maximum safe capacity` | `density > 0.45` | Safety |
| | `EMERGENCY_ALERT` (legacy) | `density >= 5.5` OR `occupancy >= 800` | Safety |
| | `ALERT: Rapid crowd surge detected …` | `flow_magnitude > 4.0` AND `density > 0.20` | Safety |
| | `SYSTEM_OVERLOAD: Multiple zones under pressure …` | `density > 0.25` AND `∀ neighbour: neighbour.density > 0.15` | Safety |
| **Warning / Stagnation** | `WARNING: Zone approaching capacity limit` | `density > 0.28` | Safety |
| | `SAFETY_WARNING tier=warning` (legacy) | `density >= 3.5` | Safety |
| | `SAFETY_WARNING tier=standstill` (legacy) | `density >= 4.0` | Safety |
| | `WARNING: Crowd stagnation detected …` | `flow_magnitude < 0.3` AND `density > 0.30` | Safety |
| **Rerouting Directives** | `REROUTE_FANS` | `density > 0.25` AND `∃ neighbour: neighbour.density < 0.12` | CrowdFlow |
| | `GATE_THROTTLE` | `zone_id.startswith("gate_")` AND `gate_in - gate_out > 40` | CrowdFlow |
| | `REROUTE_TRAFFIC` (legacy) | `growth_pct >= 25.0` AND `density >= 2.0` | CrowdFlow |
| **Flash Sales / Promotions** | `FLASH_SALE` | `density < 0.12` AND `flow_magnitude > 1.8` | Concession |
| | `FLASH_DEAL` (legacy) | `utilization < 0.25` AND `last_flash_deal > 120 s ago` | Concession |
| | `PAUSE_PROMOTIONS` | `density > 0.30` | Concession |
| | `WELCOME_DEAL` | `zone_id.startswith("gate_")` AND `gate_in > 25` AND `density < 0.18` | Concession |
| | `DYNAMIC_PRICE_UP` (legacy) | `tx_velocity >= 40 / min` (multiplier `1.15`) | Concession |
| | `TARGETED_PROMOTION` (legacy) | `intent_count >= 15` within last `60 s` | Concession |

### 3.5 CooldownManager — how rate limiting actually works

**File:** `agents/cooldown_manager.py`.

#### State machine (per `(action, zone_id)` key)

```
READY ──(should_fire = True)─────▶ ACTIVE
ACTIVE ──(condition still met)──▶ ACTIVE        # never re-fires
ACTIVE ──(mark_cleared)─────────▶ COOLDOWN
COOLDOWN ──(timer expires)──────▶ READY
```

#### Per-action cooldown table (verbatim, `cooldown_manager.py:24-36`)

```python
COOLDOWNS: dict[str, int] = {
    "CRITICAL":              45,
    "HIGH_DENSITY_WARNING":  45,
    "CROWD_SURGE":           30,
    "STAGNATION_WARNING":    30,
    "SYSTEM_OVERLOAD":       60,
    "REROUTE_FANS":          90,
    "GATE_THROTTLE":         60,
    "FLASH_SALE":            60,
    "PAUSE_PROMOTIONS":      45,
    "WELCOME_DEAL":          90,
}
DEFAULT_COOLDOWN: int = 45
```

| Action key | Cooldown (s) | Notes |
|---|---|---|
| `CRITICAL` | **45** | Highest-severity safety alert |
| `HIGH_DENSITY_WARNING` | **45** | Warning tier |
| `CROWD_SURGE` | **30** | Short — bursts may recur quickly |
| `STAGNATION_WARNING` | **30** | Same |
| `SYSTEM_OVERLOAD` | **60** | Cross-zone alert; allow time for response |
| `REROUTE_FANS` | **90** | Long — gives crowd time to actually move |
| `GATE_THROTTLE` | **60** | Operational change at gate |
| `FLASH_SALE` | **60** | Don't spam fans |
| `PAUSE_PROMOTIONS` | **45** | — |
| `WELCOME_DEAL` | **90** | Fans only enter once |
| *(any other)* | **45** | `DEFAULT_COOLDOWN` |

#### Global per-zone rate limit (`cooldown_manager.py:47`)

```python
self._zone_min_interval: float = 8.0      # seconds
```

**Any** action on the same `zone_id` is suppressed if the previous fire on that zone was less than 8 s ago, **regardless of action type**. Different zones are completely independent.

#### `should_fire(action, zone_id)` decision flow

```python
key = f"{action}:{zone_id}"
now = time.time()

# 1) Per-zone hard rate limit
if now - self._last_zone_fire.get(zone_id, 0.0) < 8.0:
    self._total_suppressed += 1
    return False

# 2) State check
state = self._states.get(key)
if state.status == "active":
    return False                                                # never re-fires while ACTIVE
if state.status == "cooldown":
    if now - state.cleared_at < COOLDOWNS.get(action, 45):
        return False                                            # still cooling down
    state.status = "active"; state.triggered_at = now           # cooldown expired → re-arm

# 3) READY or first-ever sighting → allow
self._last_zone_fire[zone_id] = now
return True
```

#### When does `mark_cleared` fire?

Triggered in the runner — `agents/agent_runner.py:144-147`:

```python
for state_key, state in list(self.cooldown_manager._states.items()):
    if state["zone_id"] == update.zone_id and state["status"] == "active":
        if state["action"] not in proposed_keys:        # condition no longer proposed this cycle
            self.cooldown_manager.mark_cleared(state["action"], state["zone_id"])
```

In other words: when a fresh `ZoneUpdate` arrives and the agent does **not** re-propose this action (because density dropped below threshold, etc.), the action is moved from ACTIVE → COOLDOWN, starting the per-action timer.

#### Practical consequences

- `CRITICAL` for `concourse_main` fires once. It stays ACTIVE until density drops below 0.45 (with hysteresis at 0.37). Then a 45-s timer starts. Within that 45 s, **even if density spikes back over 0.45, CRITICAL will not re-fire**.
- During those same 45 s, a different action on the same zone (e.g. `WARNING`) is **also** blocked unless 8 s have passed since the last zone fire (the global `_zone_min_interval` guard).
- Actions on **different zones** are entirely independent — 9 zones could each fire once per second (subject to their own per-zone 8-s rule).

---

## 4. Data Pipeline (Backend → Frontend)

### 4.1 End-to-end flow

```
[1] feeds/*.mp4
       │  cv2.VideoCapture, wrapped in asyncio.to_thread(cap.read)
       ▼
[2] vision/pipeline.py :: VisionPipeline.run_feed()
       │  PROCESS_EVERY_N_FRAMES = 3,  Semaphore(3)
       │  → process_frame()  →  YOLO + (MOG2 | YOLO) + Farneback (+ ByteTrack for gates)
       ▼
[3] core/redis_publisher.py :: ZonePublisher.publish()
       │  EMACounter (alpha=0.3, spike=2.0)
       │  density = smoothed_count / sector_total_sqm
       │  ┌─ SET    "zone:{zone_id}:latest"   (TTL = 30 s)
       │  └─ PUBLISH "zone_updates"           (JSON ZoneUpdate)
       ▼
[4] Redis pub/sub  ───  channel: "zone_updates"  ─────────────────────────
       │
       ├─ [5a] agents/agent_runner.py :: AgentRunner
       │       ZoneStateCache.update(update)
       │       SafetyAgent.evaluate() + CrowdFlowAgent.evaluate() + ConcessionAgent.evaluate()
       │       CooldownManager.should_fire()
       │       └─ PUBLISH "agent_actions"  (JSON AgentAction payload)
       │
       └─ [5b] api/websocket_handler.py :: DashboardWSHandler.run_redis_listener()
               (single Redis subscriber → fan-out to N WebSocket clients)
               ├─ "zone_updates"   →  client.send_json({"event":"zone_update",  "data": <ZoneUpdate>})
               ├─ "zone_stale"     →  client.send_json({"event":"zone_stale",   …})
               └─ "agent_actions"  →  client.send_json({"event":"agent_action", "data": <AgentAction>})
       ▼
[6] WebSockets   ──  ws://localhost:8000/ws/dashboard
       │             ws://localhost:8000/ws/fan
       ▼
[7] dashboard/index.html  /  fan-webapp/index.html
       (cards, banners, alerts, map overlays)
```

### 4.2 Redis channel reference

| Channel | Producer | Consumer | Payload |
|---|---|---|---|
| `zone_updates` | `ZonePublisher.publish()` | `AgentRunner`, `DashboardWSHandler`, `/ws/fan` | `ZoneUpdate` JSON |
| `zone_stale` | `StalenessMonitor` | `DashboardWSHandler` | `{zone_id, last_seen, stale_since}` |
| `agent_actions` | `AgentRunner._evaluate_all()` | `DashboardWSHandler`, `/ws/fan` | `AgentAction` JSON |
| `dashboard:actions` | `Orchestrator._publish_action()` | `DashboardWSHandler`, `/ws/fan` | Resolved orchestrator decision |
| `orchestrator:decisions` | `BaseAgent.publish_decision()` | `Orchestrator` | Legacy agent decisions |
| `broadcast:zone_metrics` | `crowd_simulator` / `camera_streamer` | Legacy agents | Legacy zone metric broadcast |
| `pos:transactions` | `POSSimulator` | `ConcessionAgent` | POS transactions |
| `fan:food_intent` | Fan webapp | `ConcessionAgent` | Food intent signals |
| `pos:match_clock` | `MatchClock` | All | Sim-minute (10× speed) |
| `manual:medical_emergency` | `/api/demo/trigger-medical` | `SafetyAgent` | Manual medical override |
| `sim:control` | `/api/demo/*` | Simulators | Demo control commands |
| `camera:inference_results` | `camera_streamer` (legacy) | — | Raw inference output |

### 4.3 `ZoneUpdate` schema — `core/schemas.py`

```python
class ZoneUpdate(BaseModel):
    zone_id:        str
    density:        float            # persons / m²
    raw_count:      int              # unsmoothed count from current frame
    smoothed_count: float            # EMACounter output
    flow_direction: float = 0.0      # 0–360°
    flow_magnitude: float = 0.0      # px / frame
    gate_in:        int   = 0        # cumulative IN crossings
    gate_out:       int   = 0        # cumulative OUT crossings
    source:         str              # "cv_yolo" | "cv_dense" | "cv_dense:cal@0.000234:60%" | "stale_fallback"
    timestamp:      float = Field(default_factory=time.time)
    fps:            float            # current per-zone processing FPS
```

### 4.4 `AgentAction` schema — definition vs. wire format

#### Python dataclass (`agents/base_agent.py:35-43`)

```python
@dataclass
class AgentAction:
    action:         str
    zone_id:        str
    priority:       str              # "critical" | "high" | "medium" | "low"
    detail:         str
    public_message: str | None = None
```

#### JSON payload published on `agent_actions` (`agents/agent_runner.py:154-163`)

```json
{
  "agent":          "safety_agent",
  "action":         "WARNING: Zone approaching capacity limit",
  "zone_id":        "concourse_main",
  "priority":       "high",
  "detail":         "كثافة الردهة الرئيسية (0.310 شخص/م²) تتجاوز حد التحذير 0.28 شخص/م² (ما يعادل ~124 شخصاً). المراقبة مستمرة — الاستعداد لتوجيه الحشد.",
  "public_message": "⚠️ الردهة الرئيسية يشهد ازدحاماً متزايداً. يُنصح بتجنب هذه المنطقة مؤقتاً واختيار بديل أكثر راحة.",
  "timestamp":      1730000000.0
}
```

`public_message` is included **only when not None** (lines 162-163). Any consumer (dashboard, fan app, audit stream) reads the same flat envelope.

The dashboard WS handler wraps it once more for the browser (`api/websocket_handler.py:120-121`):

```json
{ "event": "agent_action", "data": { …agent action above… } }
```

### 4.5 Frontend i18n — what's actually implemented (audit-honest)

> **Audit clarification.** The brief frames `detail` as a dual-language dictionary `{"ar": …, "en": …}`. **The current code does not implement that shape.** Both `detail` and `public_message` are **plain Arabic `str`** values; English is reconstructed on the **frontend** from structured data. Documenting what exists, not what's aspirational.

**Evidence (read-only):**

- `AgentAction.detail: str` — `agents/base_agent.py:41`.
- All agent emit-sites use Arabic f-strings:
  - `agents/safety_agent.py:153-161, 169-178, 188-198, 207-215, 232-241`
  - `agents/crowd_flow_agent.py:204-213, 226-234`
  - `agents/concession_agent.py:155-166, 175-180, 191-200`

#### Where translation actually happens

##### `fan-webapp/index.html`

- Line 608: `let lang = "ar";`
- Line 619: `function t(k) { return I18N[lang][k] || k; }`
- Line 622-635: `applyLang()` mutates `document.documentElement.lang/dir`, swaps the toggle button text, and rewrites every element with a `data-i18n` attribute.
- Lines 673-674: zone names use **structured fields** on the data object:
  ```js
  const name = lang === "ar" ? z.ar : z.en;
  const sub  = lang === "ar" ? z.tar : z.ten;
  ```
- Lines 800-816: dynamic alert text is **regenerated client-side** from structured fields (no dual-language string in the wire payload):
  ```js
  notify(msg.priority || "low", msg.public_message || msg.action, msg.timestamp);
  ...
  const tgt = d.details && d.details.suggested_target_zone || "";
  text = lang === "ar"
       ? `🚦 يُنصح بإعادة التوجيه إلى ${tgt}.`
       : `🚦 Reroute suggested to ${tgt}.`;
  ```

##### `dashboard/index.html`

- Line 769-770: `let currentLang = 'en'; function t(key) { return I18N[currentLang][key] || I18N.en[key] || key; }`
- Line 1228: `const displayAction = _translateAction(data.action || 'ACTION');` — action codes are translated via a lookup table.
- Line 1236: `<div class="card-detail">${data.detail || ''}</div>` — the `detail` text is rendered **as-is** (Arabic). The dashboard supports an Arabic UI (`dir="rtl"`) but does not translate the dynamic `detail` string.

#### Summary of the current i18n pattern

| Element | Translation source | Mechanism |
|---|---|---|
| Static UI labels (buttons, headings, legend) | Frontend `I18N[lang]` dict | `applyLang()` rewrites `data-i18n` elements |
| Zone names | Frontend `I18N.ar` + structured `z.ar` / `z.en` fields | Direct read |
| Action codes (e.g. `REROUTE_FANS`) | Frontend lookup `_translateAction` / `t(`agent_…`)` | Direct read |
| Dynamic `public_message` (toast text) | Sometimes regenerated from `details.*` structural fields; sometimes used as-is (Arabic) | `notify(..., msg.public_message || msg.action, ...)` |
| Dynamic `detail` (long explanation) | Backend Arabic only | Rendered verbatim |

#### What the dual-language pattern *would* look like (if implemented)

```python
@dataclass
class AgentAction:
    action:         str
    zone_id:        str
    priority:       str
    detail:         dict           # {"ar": "...", "en": "..."}
    public_message: dict | None = None
```

```js
// Frontend
card.querySelector('.card-detail').textContent =
    data.detail[currentLang] || data.detail.en;
```

This pattern is **not** in the code today. The current architecture chose to ship long explanatory `detail` strings only in Arabic and to recompose short alert text on the client from **structured** fields (`details.suggested_target_zone`, `details.tier`, `details.density`, …).

---

## 5. Compact Architecture Cheat-Sheet (interview prep)

- **CV stack:** YOLOv8n (`conf=0.15, imgsz=1280, classes=[0]`) + MOG2 (`history=500, varThreshold=50, warmup=50, kernel=(3,3) ELLIPSE`) + Farneback (`pyr_scale=0.5, levels=3, winsize=15`) + ByteTrack at gates (`persist=True`).
- **Mode switch:** YOLO ⇄ MOG2 at `≥ 100` detections in production (`vision/pipeline.py:55`); `≥ 40` in `tools/cv_playground.py:299` — divergence to call out.
- **Cross-calibration:** EMA `lr=0.05`, accept range `[10, 120]` YOLO detections, `min_fg_pixels=1000`, confidence saturates after 20 samples.
- **CPU protection:** `PROCESS_EVERY_N_FRAMES = 3` (≈67% inference reduction) + `asyncio.Semaphore(3)` (≤3 concurrent inferences) + `asyncio.to_thread(cap.read)`.
- **9 zones:** 3 gates + 4 stands/concourse + 1 food court + 1 activation zone — all in `config/zones.json` with `video_source` per zone.
- **Isolated test:** `python tools/cv_playground.py --zone <zone_id>` — runs the same CV stack as production, no Redis/FastAPI/agents.
- **Critical thresholds (Safety):** `density > 0.45` (CRITICAL), `flow_mag > 4.0 ∧ d > 0.20` (SURGE), `flow_mag < 0.3 ∧ d > 0.30` (STAGNATION), `d > 0.25 ∧ all neighbours > 0.15` (OVERLOAD).
- **Reroute (CrowdFlow):** `d > 0.25 ∧ ∃ neighbour < 0.12` (REROUTE_FANS); `gate_in - gate_out > 40` (GATE_THROTTLE).
- **Concession:** `d < 0.12 ∧ flow > 1.8` (FLASH_SALE); `d > 0.30` (PAUSE); `gate_in > 25 ∧ d < 0.18` (WELCOME).
- **Cooldown:** `COOLDOWNS` dict (30 / 45 / 60 / 90 s by action), global `_zone_min_interval = 8.0 s`, four-state machine READY/ACTIVE/COOLDOWN/READY.
- **Pipeline:** Camera → Pipeline → Redis (`zone_updates`) → AgentRunner → Redis (`agent_actions`) → DashboardWS → UI.
- **i18n:** flat Arabic `detail` / `public_message` strings on the wire; English UI is reconstructed on the frontend from structured `details.*` fields and a static `I18N[lang]` dictionary.

---

*Generated under a strict read-only audit. No source files were modified.*
