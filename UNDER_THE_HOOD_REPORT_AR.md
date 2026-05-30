# تقرير "تحت الغطاء" — مشروع CrowdFlow

> **المُعِدّ:** مدقق معماري للكود (Read-Only Audit)
> **الحالة:** تم إجراء فحص كامل في وضع القراءة فقط — لم يتم تعديل أي ملف.
> **النطاق:** كل ما هو موجود في المستودع الفعلي (ليس وعوداً معماريّة، بل أكواد فعلية).
> **هدف الوثيقة:** تجهيز مرشح لمقابلة هندسية تقنية عميقة — لذلك نُورد أرقاماً مطابقة للكود السطر بالسطر.

---

## 0. الخريطة المعمارية المختصرة (للسياق)

```
+---------------------------------------------------------------+
|  9 video feeds (feeds/*.mp4 | *.mov)                          |
+---------------------------------------------------------------+
              |
              v
+---------------------------------------------------------------+
| vision/pipeline.py  ::  VisionPipeline                        |
|   - YOLOv8n  (ultralytics.YOLO("yolov8n.pt"))                 |
|   - MOG2     (cv2.createBackgroundSubtractorMOG2)             |
|   - Farneback flow (cv2.calcOpticalFlowFarneback)             |
|   - ByteTrack (gate zones)                                    |
|   - asyncio.Semaphore(3)  +  PROCESS_EVERY_N_FRAMES=3         |
+---------------------------------------------------------------+
              |
              v   ZonePublisher (core/redis_publisher.py)
+---------------------------------------------------------------+
| Redis  ::  zone:{id}:latest  (SET, TTL=30s)                   |
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
|   - DashboardWSHandler  (one Redis sub → fan-out to N clients)|
|   - /ws/dashboard, /ws/fan, /api/fan/ask, /api/demo/*         |
+---------------------------------------------------------------+
              |
              v
+---------------------------------------------------------------+
| dashboard/index.html  +  fan-webapp/index.html                |
+---------------------------------------------------------------+
```

نقطتا الإقلاع: `run_demo.py` (المسار الإنتاجي المُجمَّع) و `run_all.py` (المسار التراثي مع `Orchestrator` و `CameraInferenceEngine`).

---

## 1. محرك الرؤية الحاسوبية (Computer Vision Engine) والقياسات الدقيقة

### 1.1 النموذج الفعلي

**الملف:** `vision/pipeline.py` — السطور 9-10، 49، 70.

النموذج الأساسي هو **YOLOv8-nano** عبر مكتبة `ultralytics`:

```python
from ultralytics import YOLO
...
self.model = YOLO(yolo_model_path)   # default: "yolov8n.pt"
```

النموذج هو **Single-Shot Detector** ذو خط أنبوب (backbone) من نوع CSP-Darknet مع رأس Anchor-Free، وفئة الاهتمام الوحيدة هي `class=0` (أي `person` في فهرس COCO).

النظام **هجين** (لا يعتمد على YOLO وحده) — هناك أربعة محركات رؤية تعمل بالتوازي على كل منطقة:

| المحرك | الملف | الاستخدام |
|---|---|---|
| YOLOv8n (Detection) | `vision/pipeline.py` | عدّ الأشخاص في المناطق العامة والبوابات |
| MOG2 Background Subtractor | `vision/density_estimator.py` | تقدير كثافة عند الازدحام الشديد |
| Farneback Dense Optical Flow | `vision/flow_analyzer.py` | اتجاه التدفق وحجم الحركة |
| YOLO + ByteTrack | `vision/gate_counter.py` | عدّ عبور الحاجز الافتراضي عند البوابات |

### 1.2 العتبات الرقمية الدقيقة (Confidence, imgsz, ...) — مُستخرجة من الكود

#### مناطق `general` — `vision/pipeline.py:168`

```python
results = self.model(frame, conf=0.15, classes=[0], verbose=False, imgsz=1280)
```

- `conf = 0.15` — عتبة ثقة مُنخفضة عمداً لاستلتقاط الأشخاص المحجوبين جزئياً.
- `imgsz = 1280` — دقة الإدخال (مرفوعة من 640 لاسترداد الكشف على الأشخاص البعيدين). موثّق في تعليق الكود: «recovers distant/small person detections that 640 misses».
- `classes = [0]` — حصر الفئة على `person` فقط (تقليل الضوضاء).

#### بوابات `gate` — تعدّ مرتين بإعدادات مختلفة

في `vision/pipeline.py:143` (العدّ اللحظي للبوابة):
```python
results = self.model(frame, conf=0.25, classes=[0], verbose=False)
```

وفي `vision/gate_counter.py:26`:
```python
confidence_threshold: float = 0.15
```

ثم داخل `process_frame` السطر 55-62:
```python
results = self.model.track(
    frame, persist=True, conf=self.confidence_threshold,
    classes=[0], verbose=False, imgsz=1280,
)
```
`persist=True` ضرورية للحفاظ على حالة ByteTrack بين الإطارات.

#### MOG2 (المُقدِّر الكثيف) — `vision/density_estimator.py:40-42`

```python
self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)
```

- `history = 500` — عدد الإطارات لبناء نموذج الخلفية.
- `varThreshold = 50` — عتبة المسافة Mahalanobis (قبول/رفض أن البيكسل خلفية).
- `detectShadows = False` — تجاهل الظلال (تقلل التعقيد).
- `_WARMUP_FRAMES = 50` (السطر 10) — أوّل 50 إطاراً يُرجع المُقدِّر `-1` (لم يجهز بعد).
- `min_area = 100` بكسل (السطر 28) — تخفيض من 500 لاسترداد المسافات البعيدة.
- `learning_rate = 0.005` (السطر 25).
- `calibration_factor` افتراضي = `0.005` ولكن `config/zones.json` يستخدم `0.0025` لكل المناطق العامة.
- نواة Morphology: `(3, 3) ELLIPSE` — السطر 68 (مُخفّضة من `(5,5)` لئلا تمحو الكتل الصغيرة).

العدّ النهائي:
```python
fg_pixels = cv2.countNonZero(cleaned_mask)
count = int(fg_pixels * self.calibration_factor)
```

#### عتبة الانتقال YOLO → MOG2 — `vision/pipeline.py:55, 181-198`

```python
switch_threshold: int = 100
...
if yolo_count >= self.switch_threshold:
    dense_count = dense_est.estimate(frame, roi)
    if dense_count == -1:
        raw_count = yolo_count           # source = "cv_yolo"
    else:
        raw_count = dense_count          # source = "cv_dense"
else:
    raw_count = yolo_count               # source = "cv_yolo"
```

دون 100 شخص نثق في YOLO، فوقها يبدأ الانسداد البصري (Occlusion) فننتقل إلى MOG2.

#### المُعايِر التبادلي (CrossCalibrator) — `vision/cross_calibrator.py:17-26`

```python
learning_rate    = 0.05
min_yolo_count   = 10
max_yolo_count   = 120
min_fg_pixels    = 1000
```

تحديث EMA البطيء على `calibration_factor`:
```python
observed_factor = yolo_count / fg_pixels
self._current_factor = (
    self.learning_rate * observed_factor
    + (1.0 - self.learning_rate) * self._current_factor
)
```
وفلتر الشذوذ: يُرفض كل تحديث تكون نسبته `> 100x` أو `< 0.01x` من العامل الحالي (السطر 75).
الثقة تصل إلى 1.0 بعد 20 عيّنة (`self._confidence = min(1.0, self._samples / 20.0)`).

#### Farneback Optical Flow — `vision/flow_analyzer.py:44-55`

```python
flow = cv2.calcOpticalFlowFarneback(
    self._prev_gray, gray_frame, None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)
```
عتبة الركود: `STAGNATION_THRESHOLD = 0.5` بكسل/إطار (السطر 12). أيّ مقدار حركة دونها يُعتبر `is_stagnant=True`.
زاوية الاتجاه تُحسب بـ `np.degrees(np.arctan2(mean_dy, mean_dx)) + 360) % 360`.

#### تنعيم العدّ — `core/smoothing.py` :: `EMACounter`

- `alpha = 0.3` — معامل النعومة الأسي.
- `spike_threshold = 2.0` — أي تغيّر نسبي > 200% يُرفض.
- `N_OVERRIDE = 5` — بعد 5 رفضات متتابعة تتجاوز الفلتر وتقبل القيمة الخام (لحماية نفسها من القفل عند تغيّر المشهد جذريّاً).
- حماية الصفر: `raw_count==0` و `_ema>10` → رفض حتمي (يفترض أن المشهد فعلاً غير فارغ — هذا فشل كاشف لا تغيّر مشهد).

### 1.3 رياضيات Frame Skipping (لمنع خنق المعالج)

**الملف:** `vision/pipeline.py:25, 248-294`.

```python
PROCESS_EVERY_N_FRAMES = 3
...
async def run_feed(self, zone_id, video_source, zone_type):
    cap = cv2.VideoCapture(video_source)
    frame_index = 0
    while True:
        ret, frame = await asyncio.to_thread(cap.read)
        ...
        frame_index += 1

        # Always read frames to keep video in sync, but only run heavy
        # CV inference (YOLO/MOG2/ByteTrack) every Nth frame.
        if frame_index % PROCESS_EVERY_N_FRAMES != 0:
            continue
        ...
        async with self._inference_semaphore:
            await self.process_frame(zone_id, frame, zone_type)
```

#### المنطق الرياضي خطوة بخطوة:

1. **`cap.read()` يعمل في كل إطار** — هذا حرج. لا نتجاوز قراءة الإطار، لأن ذلك يُسبب انجراف الساعة (clock drift) في الفيديوهات المُسجَّلة (يبدو الفيديو وكأنه يعمل بسرعة 3×).
2. **العدّاد `frame_index % PROCESS_EVERY_N_FRAMES`** — يُنفّذ الاستدلال (YOLO + MOG2 + ByteTrack + Farneback) فقط حين يتحقق `frame_index ≡ 0 (mod 3)`. أي إطار 3, 6, 9, 12, ...
3. **التخفيض الحسابي:** يصبح حمل الـ CPU من الاستدلال = `1/3 ≈ 33%` فقط من حمل الإطار الكامل، أي توفير **~67%** (الكود يُعلّق «reduces CPU load by ~70%»).
4. **`asyncio.to_thread(cap.read)`** — يمنع حلقة الأحداث من التجمّد أثناء قراءة OpenCV من القرص (I/O blocking → تنفيذ في thread منفصل).

#### الطبقة الثانية للحماية — `asyncio.Semaphore(3)`

`vision/pipeline.py:67`:
```python
self._inference_semaphore = asyncio.Semaphore(3)
```
حتى لو شغّلنا 9 خلاصات (zones)، **3 مناطق فقط** تستطيع تشغيل YOLO/MOG2 في اللحظة نفسها. هذا منطقيّ على جهاز MacBook Air (8 cores) كما يشير التعليق. أي طلب رابع يُحجز على `await self._inference_semaphore.__aenter__()` حتى يُحرر سَبَقه القُفل.

#### قياس الـ FPS لكل منطقة — السطور 287-290:
```python
now = time.perf_counter()
elapsed = now - prev_time
self._zone_fps[zone_id] = 1.0 / elapsed if elapsed > 0 else 0.0
```
يُحسب على الإطارات **المُعالجة فقط** (وليس كل الإطارات المقروءة).

#### إعدادات إضافية في `config/zones.json` تتحكم بنفس السلوك:
```json
"demo_settings": {
    "process_every_n_frames": 3,
    "max_concurrent_inference": 3
}
```

---

## 2. المناطق التسع واختبار الفيديو المعزول

### 2.1 الأسماء/المعرّفات الدقيقة للمناطق التسع

**المصدر الرسمي:** `config/zones.json` — هو المُصدر الذي يقرؤه `run_demo.py` و `vision/pipeline.py`.

| # | `zone_id` | `type` | `area_sqm` | `sector_total_sqm` | `capacity` | `video_source` |
|---|-----------|--------|------------|--------------------|-----------|----------------|
| 1 | `gate_main`        | gate    | 80  | 80   | 400 | `feeds/real_gate_main.mp4` |
| 2 | `gate_vip`         | gate    | 60  | 60   | 300 | `feeds/real_gate_vip.mp4` |
| 3 | `gate_south`       | gate    | 70  | 70   | 350 | `feeds/fan_plaza.mov` |
| 4 | `concourse_main`   | general | 350 | 2800 | —   | `feeds/real_concourse.mp4` |
| 5 | `stands_east`      | general | 300 | 3000 | —   | `feeds/real_stands_dense.mp4` |
| 6 | `stands_west`      | general | 300 | 3000 | —   | `feeds/stands_west.mov` |
| 7 | `stands_north`     | general | 280 | 2800 | —   | `feeds/stands_north.mov` |
| 8 | `food_court`       | general | 400 | 1600 | —   | `feeds/food_court.mp4` |
| 9 | `activation_zone`  | general | 500 | 2000 | —   | `feeds/activation_zone.mov` |

#### تفاصيل البوابات (Tripwire):
- `gate_main`: `p1=[640,0]`, `p2=[640,720]`, `in_direction="right"` (خط رأسي وسطي).
- `gate_vip`:  `p1=[0,360]`, `p2=[1280,360]`, `in_direction="down"` (خط أفقي وسطي).
- `gate_south`: `p1=[640,0]`, `p2=[640,720]`, `in_direction="right"`.

#### مفهوم Coverage Ratio (نسبة التغطية)
يستخدمه `vision/pipeline.py:151-156, 203-207`:

```python
sector_sqm     = float(cfg.get("sector_total_sqm", area_sqm))
coverage_ratio = sector_sqm / max(area_sqm, 1.0)
if coverage_ratio > 1.0:
    raw_count = int(raw_count * coverage_ratio)
    source    = f"{source}:x{coverage_ratio:.0f}"
```
المعنى: الكاميرا تُصوّر `area_sqm` فقط، لكن المنطقة الكاملة `sector_total_sqm`. لذا نُقدّر العدد على القطاع كاملاً عبر ضرب اللُقطة في النسبة. الكثافة (count/area) تظل ثابتة لأن البَسَط والمَقام يُضربان بنفس النسبة.

> **ملاحظة:** يوجد ملف `config.py` يحتوي قاموساً مختلفاً اسمه `ZONES` فيه 10 مناطق (7 داخلية + 3 خارجية: `zone_north_gate`، `zone_south_gate`، `zone_east_stand`، `zone_west_stand`، `zone_north_concourse`، `zone_south_concourse`، `zone_pitch_view`، `zone_activation_1`، `zone_activation_2`، `zone_north_parking`). هذا التعريف **تراثي**؛ يستخدمه نظام الوكلاء القديم (`agents/orchestrator.py`، `agents/crowd_flow_agent.py` للإحالات الخارجية، `simulation/crowd_simulator.py`) وكذلك واجهة `backend/main.py` لقائمة المناطق الافتراضية. أمّا المسار الإنتاجي للرؤية الحاسوبية فيستهلك حصراً `config/zones.json` (المناطق التسع أعلاه).

### 2.2 آلية اختبار خلاصة فيديو واحدة معزولة (CRUCIAL)

#### الإجابة المباشرة: نعم — توجد أداة CLI مخصّصة لذلك:
**الأداة:** `tools/cv_playground.py`

**الفلسفة:** هذه الأداة تُشغّل **نفس مكوّنات الإنتاج** (YOLO، DenseEstimator، CrossCalibrator، OpticalFlowAnalyzer، GateTripwireCounter، EMACounter) مع **صفر اعتمادية على Backend** (لا Redis، لا FastAPI، لا WebSockets، لا وكلاء).

#### الطريقة 1 — التحميل التلقائي عبر `--zone` (الأسهل والأقوى)

```bash
python tools/cv_playground.py --zone concourse_main
```

ما يحدث داخلياً (`tools/cv_playground.py:139-183`):
1. تفتح الأداة `config/zones.json` وتقرأ المنطقة المطلوبة.
2. تُرَتِّب جميع المعطيات تلقائياً من ملف الكونفِغ:
   - `args.video` ← `zcfg["video_source"]`
   - `args.mode`  ← `zcfg["type"]` (`"general"` أو `"gate"`)
   - `args.area_sqm` ← `zcfg["area_sqm"]`
   - `args.roi`   ← `json.dumps(zcfg["roi"])`
   - `args.calibration_factor` ← `zcfg["calibration_factor"]` (للمناطق العامة)
   - `args.tripwire` ← `json.dumps([tw["p1"], tw["p2"]])` (للبوابات)
   - `args.in_direction` ← `zcfg["in_direction"]`

**هذا هو الربط الفعلي:** ملف `config/zones.json` هو خريطة الفيديوهات-↔-المناطق. لا توجد ملفات منفصلة، ولا متغيرات بيئة. كل شيء في JSON واحد.

#### الطريقة 2 — التجاوز اليدوي

```bash
# منطقة عامة
python tools/cv_playground.py feeds/real_concourse.mp4 \
    --mode general \
    --roi '{"x":85,"y":42,"w":1110,"h":636}' \
    --area-sqm 400 \
    --calibration-factor 0.0025

# بوابة
python tools/cv_playground.py feeds/real_gate_main.mp4 \
    --mode gate \
    --tripwire '[[640,0],[640,720]]' \
    --in-direction right
```

#### خيارات مهمة:
- `--headless` — يُلغي واجهة OpenCV ويطبع إحصائيات نصّية فقط (مفيد على الخوادم).
- `--export-csv path.csv` — يُسجّل المقاييس لكل إطار (yolo_count, dense_count, fg_pixels, cal_factor, smoothed, density, flow_dir, flow_mag، إلخ).
- `--slow` — وضع خطوة بخطوة (يطلب ضغطة مفتاح بين كل إطار).
- `--conf 0.15 --imgsz 1280` — تجاوز عتبة الثقة ودقة الإدخال.
- مفاتيح أثناء التشغيل: `SPACE` للإيقاف المؤقت، `Q` للخروج، `S` لحفظ صورة الإطار، `N` للإطار التالي عند الإيقاف.

#### آلية المعالجة الكاملة (`process_frame_general` السطور 277-337)

```python
1. results       = yolo(frame, conf=0.15, classes=[0], imgsz=1280)
   yolo_count   = len(results[0].boxes)
2. dense_count  = dense_estimator.estimate(frame, (x,y,w,h))
   fg_pixels    = dense_estimator.get_last_fg_pixels()
3. cal_factor   = cross_calibrator.update(yolo_count, fg_pixels)
4. # auto-switch (NB: hardcoded 40 here, vs 100 in production pipeline.py)
   if yolo_count >= 40 and dense_count >= 0:
       chosen_count = dense_count;  mode = "DENSE (MOG2)"
   else:
       chosen_count = yolo_count;   mode = "YOLO"
5. smoothed     = ema.update(chosen_count)   # alpha=0.3, spike=2.0
6. flow_results = flow_analyzer.analyze(frame)
7. density      = smoothed / area_sqm
```

> **اكتشاف صغير لكنه مهم:** عتبة الانتقال في الـ playground ثابتة عند **40** (`switch_threshold = 40` السطر 299) بينما في الإنتاج هي **100** (`vision/pipeline.py:55`). هذا تباين قد يؤدي لسلوك مختلف عند التصحيح.

#### واجهة العرض الرباعية (4-pane) للمناطق العامة:
- أعلى يسار: `draw_yolo_overlay` — صناديق YOLO وتعليقات الثقة.
- أعلى يمين:  `draw_mog2_overlay` — قناع MOG2 الأبيض-أسود مع `fg_pixels`، `cal_factor`، علامة `WARMING UP`.
- أسفل يسار: `draw_flow_overlay` — سهم الاتجاه (أحمر إذا `is_stagnant`، أخضر خلاف ذلك).
- أسفل يمين: `draw_stats_panel` — لوحة نصّية بكل المقاييس.

#### للبوابات (`process_frame_gate` السطور 339-358):
- يُشغّل `gate_counter.process_frame(frame)` (ByteTrack + Tripwire).
- يحسب `gate_in - gate_out = net_occupancy`.

---

## 3. وكلاء الذكاء الاصطناعي ومنطق الفترة الباردة (Cooldown)

### 3.1 وكيل السلامة (SafetyAgent) — `agents/safety_agent.py`

#### العتبات الفعلية المُستخدمة في `evaluate(update: ZoneUpdate)`

**الكتل النصّية الثابتة (small-venue calibrated)** — السطور 51-75:

| الثابت | القيمة | الإجراء (action) | الأولوية |
|---|---|---|---|
| `SMALL_DENSITY_CRITICAL`        | `0.45` p/m² | **CRITICAL: Zone at maximum safe capacity** | `critical` |
| `SMALL_DENSITY_CRITICAL_CLEAR`  | `0.37` p/m² | (هيستيريسيس فقط) | — |
| `SMALL_DENSITY_WARNING`         | `0.28` p/m² | **WARNING: Zone approaching capacity limit** | `high` |
| `SMALL_DENSITY_WARNING_CLEAR`   | `0.18` p/m² | (هيستيريسيس فقط) | — |
| `SMALL_FLOW_SURGE`              | `4.0` px/frame | **ALERT: Rapid crowd surge detected** (يشترط أيضاً `d > 0.20`) | `critical` |
| `SMALL_FLOW_SURGE_MIN_DENSITY`  | `0.20` p/m² | شرط مُكمّل لـ FLOW_SURGE | — |
| `SMALL_DENSITY_OVERLOAD`        | `0.25` p/m² | **SYSTEM_OVERLOAD** (يشترط أن جميع الجيران `> SMALL_ADJACENT_PRESSURE`) | `critical` |
| `SMALL_ADJACENT_PRESSURE`       | `0.15` p/m² | عتبة جار مُجهَد | — |
| `SMALL_STAGNATION_MIN_DENSITY`  | `0.30` p/m² | **WARNING: Crowd stagnation** (يشترط أيضاً `mag < 0.3`) | `high` |

#### المنطق المُكتوب حرفياً (السطور 147-241):

```python
# كثافة حرجة (التوجيهات العاجلة)
if d > SMALL_DENSITY_CRITICAL:           # > 0.45
    actions.append(AgentAction(
        action="CRITICAL: Zone at maximum safe capacity",
        priority="critical",
        detail=f"كثافة {_zone_ar(zid)} ({d:.3f} شخص/م²) تتجاوز الحد الحرج "
               f"{SMALL_DENSITY_CRITICAL} شخص/م²..."
    ))
elif d > SMALL_DENSITY_WARNING:           # > 0.28
    actions.append(AgentAction(
        action="WARNING: Zone approaching capacity limit",
        priority="high", ...
    ))

# قَفزة تدفق (Crowd Surge)
if mag > SMALL_FLOW_SURGE and d > SMALL_FLOW_SURGE_MIN_DENSITY:    # > 4.0 و > 0.20
    actions.append(AgentAction(
        action="ALERT: Rapid crowd surge detected — abnormal movement speed in occupied zone",
        priority="critical", ...
    ))

# ركود (Stagnation/Bottleneck)
if mag < 0.3 and d > SMALL_STAGNATION_MIN_DENSITY:                 # < 0.3 و > 0.30
    actions.append(AgentAction(
        action="WARNING: Crowd stagnation detected — possible bottleneck forming",
        priority="high", ...
    ))

# تحميل عام (System overload)
if d > SMALL_DENSITY_OVERLOAD and self._zone_cache is not None:    # > 0.25
    adjacent = self._zone_cache.get_adjacent_zones(zid)
    if adjacent and all(a.density > SMALL_ADJACENT_PRESSURE for a in adjacent):  # كل الجيران > 0.15
        actions.append(AgentAction(
            action="SYSTEM_OVERLOAD: Multiple zones under pressure — consider venue-wide intervention",
            priority="critical", ...
        ))
```

#### العتبات التراثية (المسار `broadcast:zone_metrics`) — السطور 39-43

| الثابت | القيمة | الإجراء |
|---|---|---|
| `DENSITY_CRUSH_RISK` (`CRUSH_RISK_DENSITY`) | `5.5` p/m² | `EMERGENCY_ALERT` (priority=0) |
| `DENSITY_STANDSTILL` (`STANDSTILL_DENSITY`) | `4.0` p/m² | `SAFETY_WARNING tier=standstill` (priority=1) |
| `DENSITY_WARNING` (`CONGESTION_HIGH`) | `3.5` p/m² | `SAFETY_WARNING tier=warning` (priority=1) |
| `COUNT_EMERGENCY` | `800` شخصاً | بديل لمحفّز الطوارئ بالعدد المطلق |
| `CLEAR_FACTOR` | `0.80` | شرط الإلغاء = `density < 5.5×0.80 AND occupancy < 800×0.80` |

`config.py:218-219` يحوي أيضاً:
- `SAFETY_MAX_DENSITY  = 6.0` p/m² (محفّز الإخلاء الصارم)
- `SAFETY_MAX_FLOW_RATE = 80` شخص/دقيقة عبر بوابة
- `SAFETY_EMERGENCY_COUNT = 800`

### 3.2 وكيل تدفق الحشد (CrowdFlowAgent) — `agents/crowd_flow_agent.py`

#### العتبات في `evaluate()` (small-venue) — السطور 45-51

| الثابت | القيمة | الإجراء |
|---|---|---|
| `SMALL_REROUTE_DENSITY`    | `0.25` p/m² | **REROUTE_FANS** (priority=`high`) |
| `SMALL_ADJACENT_FREE`      | `0.12` p/m² | شرط: المنطقة المجاورة يجب أن تكون `< 0.12` ليُعتبر أنّ لديها سعة |
| `SMALL_GATE_NET_IMBALANCE` | `40` شخصاً | **GATE_THROTTLE** عند `gate_in - gate_out > 40` |

```python
# توجيه إعادة الحركة (Rerouting)
if d > SMALL_REROUTE_DENSITY and self._zone_cache is not None:    # > 0.25
    for adj in self._zone_cache.get_adjacent_zones(zid):
        if adj.density < SMALL_ADJACENT_FREE:                     # < 0.12
            actions.append(AgentAction(action="REROUTE_FANS", priority="high", ...))
            break

# خنق البوابة (Gate Throttle)
if zid.startswith("gate_"):
    net = update.gate_in - update.gate_out
    if net > SMALL_GATE_NET_IMBALANCE:                            # > 40
        actions.append(AgentAction(action="GATE_THROTTLE", priority="high", ...))
```

#### مَنطق التراث/التنبؤ — السطور 36-40
- `ROLLING_WINDOW = 30` (آخر 30 قراءة لكل منطقة).
- `RECENT_WINDOW = 5` و `OLDER_WINDOW = 5` (مقارنة آخر 5 مع الـ 5 السابقة).
- `GROWTH_THRESHOLD = 25.0%` — معدل النمو الذي يُحفّز `REROUTE_TRAFFIC`.
- `PREDICTION_MINUTES = 5.0` — أُفُق التنبؤ الخطّي.

```python
def _compute_growth(self, zone_id):
    items     = list(buf)
    older     = items[-(RECENT+OLDER):-RECENT]
    recent    = items[-RECENT:]
    avg_older  = sum(older)/len(older)
    avg_recent = sum(recent)/len(recent)
    return ((avg_recent - avg_older) / avg_older) * 100.0
```

`config.py:222-223` يحدد بدلائل التنبيه أيضاً:
- `GROWTH_RATE_WARNING  = 15.0` ٪
- `GROWTH_RATE_CRITICAL = 30.0` ٪

### 3.3 وكيل المبيعات (ConcessionAgent) — `agents/concession_agent.py`

#### العتبات في `evaluate()` (small-venue) — السطور 50-64

| الثابت | القيمة | الإجراء |
|---|---|---|
| `SMALL_FLASH_MAX_DENSITY` | `0.12` p/m² | **FLASH_SALE** (priority=`low`) — يُشترط معه: |
| `SMALL_FLASH_MIN_FLOW`    | `1.8` px/frame | تدفق المشاة فوق ضوضاء الخلفية |
| `SMALL_PAUSE_DENSITY`     | `0.30` p/m² | **PAUSE_PROMOTIONS** (priority=`medium`) |
| `SMALL_GATE_INFLUX`       | `25` شخصاً (`gate_in`) | **WELCOME_DEAL** — يُشترط: |
| `SMALL_GATE_MAX_DENSITY`  | `0.18` p/m² | البوابة لم تختنق بعد |

```python
# عرض خاطف (شَخص يَعبر، الحضور خفيف)
if d < SMALL_FLASH_MAX_DENSITY and mag > SMALL_FLASH_MIN_FLOW:    # < 0.12 و > 1.8
    actions.append(AgentAction(action="FLASH_SALE", priority="low", ...))

# تعليق العروض (المنطقة مزدحمة)
elif d > SMALL_PAUSE_DENSITY:                                      # > 0.30
    actions.append(AgentAction(action="PAUSE_PROMOTIONS", priority="medium", ...))

# عرض ترحيبي (تدفق مشجعين قادم، البوابة لم تنسد)
if zid.startswith("gate_") and update.gate_in > SMALL_GATE_INFLUX \
        and d < SMALL_GATE_MAX_DENSITY:                            # > 25 و < 0.18
    actions.append(AgentAction(action="WELCOME_DEAL", priority="low", ...))
```

#### العتبات التراثية (POS / Intent) — السطور 40-44

| الثابت | القيمة | الإجراء |
|---|---|---|
| `HIGH_VELOCITY_THRESHOLD` | `40` معاملة/دقيقة | `DYNAMIC_PRICE_UP` (مضاعف `1.15`) |
| `VELOCITY_WINDOW`         | `60.0` ثانية | نافذة قياس السرعة |
| `LOW_TRAFFIC_UTILIZATION` | `0.25` (نسبة استخدام) | شرط FLASH_DEAL |
| `FLASH_DEAL_COOLDOWN`     | `120.0` ثانية | فترة باردة محلية لكل منطقة |
| `INTENT_THRESHOLD`        | `15` نية طعام | يحفّز `TARGETED_PROMOTION` |
| `INTENT_WINDOW`           | `60.0` ثانية | نافذة تجميع النيّات |

كذلك إلغاء `surge_active` يحدث عند `velocity < HIGH_VELOCITY_THRESHOLD * 0.7` (=28 معاملة/دقيقة) — هيستيريسيس.

### 3.4 جدول مُجمَّع: من-أين-إلى-أين بدقّة

| التصنيف | الإجراء | الشرط الرياضي الكامل | وكيل |
|---|---|---|---|
| **الحرج (Critical / Overload)** | CRITICAL | `density > 0.45` | Safety |
| | EMERGENCY_ALERT (legacy) | `density >= 5.5` OR `occupancy >= 800` | Safety |
| | ALERT crowd surge | `flow_mag > 4.0` AND `density > 0.20` | Safety |
| | SYSTEM_OVERLOAD | `density > 0.25` AND `all(neighbour.density > 0.15)` | Safety |
| **التحذيرات (Warning / Stagnation)** | WARNING capacity | `density > 0.28` | Safety |
| | SAFETY_WARNING tier=warning (legacy) | `density >= 3.5` | Safety |
| | SAFETY_WARNING tier=standstill (legacy) | `density >= 4.0` | Safety |
| | WARNING stagnation | `flow_mag < 0.3` AND `density > 0.30` | Safety |
| **التوجيهات (Rerouting)** | REROUTE_FANS | `density > 0.25` AND `∃ adj. neighbour.density < 0.12` | CrowdFlow |
| | GATE_THROTTLE | `zone.startswith("gate_")` AND `gate_in - gate_out > 40` | CrowdFlow |
| | REROUTE_TRAFFIC (legacy) | `growth >= 25.0%` AND `density >= 2.0` | CrowdFlow |
| **العروض (Flash Sales / Deals)** | FLASH_SALE | `density < 0.12` AND `flow_mag > 1.8` | Concession |
| | FLASH_DEAL (legacy) | `utilization < 0.25` AND `last_deal > 120s ago` | Concession |
| | PAUSE_PROMOTIONS | `density > 0.30` | Concession |
| | WELCOME_DEAL | `zone.startswith("gate_")` AND `gate_in > 25` AND `density < 0.18` | Concession |
| | DYNAMIC_PRICE_UP (legacy) | `tx_velocity >= 40 /min` (×1.15) | Concession |
| | TARGETED_PROMOTION (legacy) | `intent_count >= 15 in 60s` | Concession |

### 3.5 CooldownManager — التطبيق الفعلي لتحديد المعدل (Rate Limiting)

**الملف:** `agents/cooldown_manager.py`.

#### آلة الحالة (State Machine)
```
READY ──(should_fire=True)──▶ ACTIVE
ACTIVE ──(should_fire=False, ضمن المقترحات)──▶ ACTIVE  (لا إعادة إطلاق)
ACTIVE ──(mark_cleared)──▶ COOLDOWN
COOLDOWN ──(انقضاء المؤقت)──▶ READY
```

#### قاموس فترات الانتظار `COOLDOWNS` بالأرقام الدقيقة (السطور 24-35):

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

#### الحد العالمي لكل منطقة (Per-Zone Rate Limit) — السطر 47:
```python
self._zone_min_interval: float = 8.0   # ثوانٍ
```
أي إجراء جديد لنفس `zone_id` لن يُطلق إن مرّ < **8 ثوانٍ** من إطلاق سابق لنفس المنطقة (مهما اختلف نوع الإجراء).

#### `should_fire(action, zone_id)` — منطق القرار (السطور 52-112)

```python
key = f"{action}:{zone_id}"
now = time.time()

# 1) حد المنطقة (8 ثوانٍ بين أي إجراءَين على نفس zone)
if now - self._last_zone_fire.get(zone_id, 0.0) < 8.0:
    self._total_suppressed += 1
    return False

# 2) فحص الحالة:
state = self._states.get(key)
if state.status == "active":      return False                      # ACTIVE → لا إعادة إطلاق
if state.status == "cooldown":
    elapsed = now - state.cleared_at
    if elapsed < COOLDOWNS.get(action, 45):  return False           # ما زال يبرد
    state.status = "active"; state.triggered_at = now               # COOLDOWN انتهى → نُعيد التفعيل

# (إذا كانت READY، أو مفتاحٌ جديد، أو خرج من COOLDOWN)
self._last_zone_fire[zone_id] = now
return True
```

#### `mark_cleared(action, zone_id)` — متى يُستدعى؟

في `agents/agent_runner.py:144-147`:
```python
for state_key, state in list(self.cooldown_manager._states.items()):
    if state["zone_id"] == update.zone_id and state["status"] == "active":
        if state["action"] not in proposed_keys:        # الشرط لم يعد مقترَحاً هذه الدورة
            self.cooldown_manager.mark_cleared(state["action"], state["zone_id"])
```
أي عندما يأتي `ZoneUpdate` جديد ولم يعد الوكيل يقترح هذا الإجراء (لأن الكثافة هبطت تحت العتبة)، نَعتبر الشرط قد زال ونبدأ المؤقت الخاص بالإجراء (`COOLDOWNS[action]`).

#### نتائج عملية مهمة:
- إنذار `CRITICAL` لمنطقة `concourse_main` يعمل مرة واحدة، ثم يبقى ACTIVE حتى تنخفض الكثافة دون 0.45؛ بعد الانخفاض يبدأ عدّ `45s`. خلال هذه 45 ثانية، حتى لو ارتدت الكثافة مجدداً، **لن** يُعاد إطلاق التنبيه (هذا صريح في الكود).
- لكن، خلال نفس 45 ثانية، إجراء **مختلف** على نفس المنطقة (مثل `WARNING`) لن يُطلق أيضاً إذا كان آخر إطلاق حدث منذ < 8 ثوانٍ (شَرَك حدّ المنطقة).
- إجراءات على **مناطق مختلفة** تعمل بشكل مستقل تماماً.

---

## 4. خط أنابيب البيانات (من الكاميرا إلى الواجهة)

### 4.1 المسار الكامل خطوة بخطوة

```
[1] feeds/*.mp4
       │  cv2.VideoCapture (asyncio.to_thread)
       ▼
[2] vision/pipeline.py :: VisionPipeline.run_feed()
       │  PROCESS_EVERY_N_FRAMES=3, Semaphore(3)
       │  → process_frame()  →  YOLO + (MOG2 | YOLO) + Farneback
       ▼
[3] core/redis_publisher.py :: ZonePublisher.publish()
       │  EMACounter(alpha=0.3, spike=2.0)
       │  density = smoothed_count / sector_total_sqm
       │  ┌─ SET   "zone:{zone_id}:latest" (TTL=30s)
       │  └─ PUBLISH "zone_updates" (JSON ZoneUpdate)
       ▼
[4] Redis pub/sub  ───── channel: "zone_updates" ─────────────────────────────────────
       │
       ├─ [5a] agents/agent_runner.py :: AgentRunner
       │       ZoneStateCache.update(update)
       │       safety_agent.evaluate()  +  crowd_flow_agent.evaluate()  +  concession_agent.evaluate()
       │       CooldownManager.should_fire()
       │       └─ PUBLISH "agent_actions"  (JSON payload)
       │
       └─ [5b] api/websocket_handler.py :: DashboardWSHandler.run_redis_listener()
               (مُشترك واحد لكل العملاء — اقتصاد كبير في الاتصالات)
               ├─ caches "zone_updates"  →  client.send_json({"event":"zone_update", "data":...})
               ├─ "zone_stale"           →  client.send_json({"event":"zone_stale", ...})
               └─ "agent_actions"        →  client.send_json({"event":"agent_action", "data":...})
       ▼
[6] WebSocket  ──  ws://localhost:8000/ws/dashboard
       │       └──  ws://localhost:8000/ws/fan
       ▼
[7] dashboard/index.html  /  fan-webapp/index.html
       (يرسم البطاقات، التنبيهات، الخرائط)
```

### 4.2 قنوات Redis الرسمية

| القناة | المُنتِج | المُستهلِك | الحمولة |
|---|---|---|---|
| `zone_updates`         | `ZonePublisher.publish()` | `AgentRunner`, `DashboardWSHandler` | `ZoneUpdate` JSON |
| `zone_stale`           | `StalenessMonitor`         | `DashboardWSHandler` | `{zone_id, last_seen, stale_since}` |
| `agent_actions`        | `AgentRunner._evaluate_all()` | `DashboardWSHandler`, `/ws/fan` | `AgentAction` JSON |
| `dashboard:actions`    | `Orchestrator._publish_action()` | `DashboardWSHandler`, `/ws/fan` | قرار مُسوّى |
| `orchestrator:decisions` | `BaseAgent.publish_decision()` | `Orchestrator` | قرارات وكلاء التراث |
| `broadcast:zone_metrics` | `crowd_simulator` / `camera_streamer` | الوكلاء التراثيون | بَثّ مقاييس قديم |
| `pos:transactions`     | `POSSimulator` | `ConcessionAgent` | معاملات نقطة البيع |
| `fan:food_intent`      | (واجهة المعجب) | `ConcessionAgent` | نوايا الطعام |
| `pos:match_clock`      | `MatchClock` | الكل | الدقيقة الحالية للمباراة (×10) |
| `manual:medical_emergency` | `/api/demo/trigger-medical` | `SafetyAgent` | تجاوز يدوي للطوارئ الطبية |
| `sim:control`          | `/api/demo/*`     | المحاكيات | أوامر العرض التوضيحي |
| `camera:inference_results` | `camera_streamer` (legacy) | — | ناتج خام للاستدلال |

### 4.3 المخطط الدقيق لكائن `ZoneUpdate` — `core/schemas.py`

```python
class ZoneUpdate(BaseModel):
    zone_id:        str
    density:        float            # شخص/م²
    raw_count:      int              # عدّ خام لهذا الإطار
    smoothed_count: float            # ناتج EMACounter
    flow_direction: float = 0.0      # 0–360°
    flow_magnitude: float = 0.0      # px/frame
    gate_in:        int   = 0        # عبور تراكمي داخل
    gate_out:       int   = 0        # عبور تراكمي خارج
    source:         str              # "cv_yolo" | "cv_dense" | "cv_dense:cal@0.000234:60%" | "stale_fallback"
    timestamp:      float = Field(default_factory=time.time)
    fps:            float            # الأطر/ثانية الحالي للمنطقة
```

### 4.4 المخطط الدقيق لكائن `AgentAction`

#### تعريف dataclass — `agents/base_agent.py:35-43`:
```python
@dataclass
class AgentAction:
    action:         str
    zone_id:        str
    priority:       str              # "critical" | "high" | "medium" | "low"
    detail:         str
    public_message: str | None = None
```

#### الحمولة JSON الفعلية المنشورة على `agent_actions` — `agents/agent_runner.py:154-163`:

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

(`public_message` اختيارية — تُضاف فقط إذا كانت `not None` — السطور 162-163.)

### 4.5 الحقيقة حول دعم التدويل (i18n) — تنبيه دقيق

> **الملاحظة:** الطلب أَفْتَرَض أن الحقل `detail` مُنفَّذ كقاموس ثنائي اللغة `{"ar": ..., "en": ...}`. **هذا غير دقيق على المستودع الفعلي.** أُسجّله بأمانة لأن وضع المراجعة "Read-Only" يقتضي التزام الحقيقة لا الادّعاء.

#### ما هو موجود فعلاً في الكود:

1. **في الـ Backend / Agents:** `detail` و `public_message` كلاهما **سلسلة نصية عربية مُجرَّدة (str)**، لا قاموس ثنائي. تَحقُّق:
   - `agents/safety_agent.py:153-161, 169-178, 188-198`: `detail=f"كثافة {_zone_ar(zid)} ... شخص/م²..."` (نص عربي صريح).
   - `agents/crowd_flow_agent.py:204-213`: نفس النمط.
   - `agents/concession_agent.py:155-166`: نفس النمط.
   - تعريف `AgentAction.detail: str` (وليس `dict`).

2. **التدويل الفعلي يحدث على جانب العميل (Frontend i18n):**
   - **fan-webapp/index.html** (السطور 608, 619, 622-635): متغيّر `let lang = "ar"` ودالّة `t(k) = I18N[lang][k] || k`. التبديل بزر `toggleLang()`. الترجمة تُطبَّق على **النصوص الثابتة** فقط (تسميات الأزرار، أسماء المناطق).
     - الأسماء العربية للمناطق مُعرَّفة في كائن `I18N` على الواجهة، وكذلك من خلال خاصيتي `z.ar / z.en` و `z.tar / z.ten`:
       ```js
       const name = lang === "ar" ? z.ar : z.en;
       const sub  = lang === "ar" ? z.tar : z.ten;
       ```
   - أمّا حقل `detail`/`public_message` الديناميكي القادم من `agent_actions`، فيُعرض كما هو من الـ Backend (السطور 790, 800-816 في `fan-webapp`):
     ```js
     notify(msg.priority || "low", msg.public_message || msg.action, msg.timestamp);
     // ...
     text = lang === "ar"
        ? `🚦 يُنصح بإعادة التوجيه إلى ${tgt}.`
        : `🚦 Reroute suggested to ${tgt}.`;
     ```
     أي أن الترجمة الإنجليزية لبعض الأنواع تُعاد إنشاؤها على الواجهة من **الحقول البنيوية** (`details.suggested_target_zone`)، **لا من قاموس ثنائي في الحمولة**.
   - **dashboard/index.html** السطر 1236:
     ```html
     <div class="card-detail">${data.detail || ''}</div>
     ```
     يُعرض النص العربي حرفياً (لا توجد ترجمة هنا — لوحة التحكم الإدارية مُعَدَّة للعرض ثنائي الاتجاه عبر تبديل `dir/rtl` على المستند، لكن نص `detail` يبقى عربياً).

#### مخطط الحقل `detail` كما لو كان مُصمَّماً ثنائي اللغة (للمقابلة):

إذا أُريد تنفيذ ذلك مستقبلاً (نمط التصميم المتوقع)، يكون:
```python
@dataclass
class AgentAction:
    action:   str
    zone_id:  str
    priority: str
    detail:   dict          # {"ar": "...", "en": "..."}
    public_message: dict | None = None     # {"ar": "...", "en": "..."}
```
ثم في الـ Frontend:
```js
card.querySelector('.card-detail').textContent = data.detail[currentLang] || data.detail.en;
```
لكن **هذا غير مُنفَّذ في الكود الحالي**.

#### العمارة الحالية بتلخيصها (الواقعية):
- الـ Backend يَكتب نصاً عربياً واحداً.
- الـ Frontend يدعم تبديل لغة واجهة المستخدم (التسميات الثابتة + أسماء المناطق + بعض الرسائل الإنشائية)، لكن نصوص `detail` الديناميكية تَعرض كما هي.
- التدويل الفعلي للحَمولات الديناميكية اعتمد على بنية **بيانات منظَّمة** (`details.suggested_target_zone`، `details.tier`، `details.density`) يُعيد الـ Frontend تركيب النص منها بالعربية أو الإنجليزية محلياً، بدلاً من قاموس `{"ar","en"}` في الحمولة نفسها.

---

## 5. خلاصة معمارية مُختصرة (لتذكير المرشح في المقابلة)

- **CV:** YOLOv8n (`conf=0.15, imgsz=1280, classes=[0]`) + MOG2 (`history=500, varThreshold=50, warmup=50, kernel=(3,3)`) + Farneback (`pyr_scale=0.5, levels=3, winsize=15`) + ByteTrack للبوابات (`persist=True`).
- **التحويل:** YOLO ⇄ MOG2 عند `>= 100` كشف (الإنتاج)؛ في playground `>= 40`. كروس-كاليبريشن EMA (`lr=0.05, range=10..120, min_fg=1000`).
- **حماية المعالج:** `PROCESS_EVERY_N_FRAMES = 3` (~67% توفير) + `asyncio.Semaphore(3)` (بحدّ أقصى 3 مناطق متزامنة) + `asyncio.to_thread(cap.read)`.
- **9 مناطق:** 3 بوابات + 4 مدرجات/ردهات + 1 مأكولات + 1 منطقة فعّاليات. كلها في `config/zones.json` مع `video_source` لكلٍّ منها.
- **الاختبار المعزول:** `python tools/cv_playground.py --zone <zone_id>` يُشغّل نفس مكوّنات الإنتاج بلا Backend.
- **العتبات الحرجة:** Critical=`density>0.45`، Surge=`mag>4.0 ∧ d>0.20`، Stagnation=`mag<0.3 ∧ d>0.30`، Overload=`d>0.25 ∧ all neighbours>0.15`.
- **الإحالات:** REROUTE_FANS=`d>0.25 ∧ ∃ neighbour<0.12`، GATE_THROTTLE=`net_in>40`.
- **العروض:** FLASH_SALE=`d<0.12 ∧ flow>1.8`، PAUSE_PROMOTIONS=`d>0.30`، WELCOME_DEAL=`gate_in>25 ∧ d<0.18`.
- **Cooldown:** قاموس `COOLDOWNS` (30/45/60/90 ثانية حسب نوع الإجراء)، حدّ منطقة عام `8s`، آلة حالة 4 حالات.
- **خط الأنابيب:** Camera → Pipeline → Redis (`zone_updates`) → AgentRunner → Redis (`agent_actions`) → DashboardWS → UI.
- **التدويل:** ليس قاموساً ثنائياً في الحمولة؛ هو نصّ عربيّ في الـ Backend مع طبقة I18N على الـ Frontend.

---

*تمت كتابة هذا التقرير بناءً على فحص قراءة-فقط لمستودع `CrowdFlow`. لم يُعدَّل أي ملف من الكود الأصلي.*
