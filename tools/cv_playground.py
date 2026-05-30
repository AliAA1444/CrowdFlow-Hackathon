"""
CrowdFlow CV Playground — isolated CV testing tool.

Runs the same vision components as the production pipeline against a video
file with zero backend dependencies (no Redis, FastAPI, WebSockets, agents).

Usage examples
--------------
# General zone (density estimation + optical flow)
python tools/cv_playground.py feeds/real_concourse.mp4 --mode general \
    --roi '{"x":85,"y":42,"w":1110,"h":636}' --area-sqm 400

# Gate zone (ByteTrack + tripwire counting)
python tools/cv_playground.py feeds/real_gate_main.mp4 --mode gate \
    --tripwire '[[640,0],[640,720]]' --in-direction right

# Quick test with auto-detected settings from zones.json
python tools/cv_playground.py --zone concourse_main

# Headless mode (no GUI, just prints stats)
python tools/cv_playground.py feeds/real_concourse.mp4 --mode general --headless
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cv_playground",
        description="CrowdFlow isolated CV testing environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to MP4 video file",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Zone ID from config/zones.json. Auto-loads all settings for that zone.",
    )
    parser.add_argument(
        "--mode",
        choices=["general", "gate"],
        default="general",
        help="Pipeline mode (default: general)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help='ROI JSON string, e.g. \'{"x":0,"y":0,"w":640,"h":480}\'',
    )
    parser.add_argument(
        "--area-sqm",
        type=float,
        default=400.0,
        dest="area_sqm",
        help="Zone area in square metres (default: 400)",
    )
    parser.add_argument(
        "--calibration-factor",
        type=float,
        default=None,
        dest="calibration_factor",
        help="Initial MOG2 calibration factor (default: 0.0025)",
    )
    parser.add_argument(
        "--tripwire",
        type=str,
        default=None,
        help='Tripwire JSON string, e.g. \'[[640,0],[640,720]]\'',
    )
    parser.add_argument(
        "--in-direction",
        choices=["up", "down", "left", "right"],
        default="down",
        dest="in_direction",
        help="Gate IN direction (default: down)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="YOLO confidence threshold (default: 0.15)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO input resolution (default: 1280)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Skip OpenCV GUI — print stats to stdout only",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        dest="export_csv",
        help="Path to write per-frame metrics CSV",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        default=False,
        help="Wait for keypress between frames (step-through mode)",
    )
    parser.add_argument(
        "--no-yolo-overlay",
        action="store_false",
        dest="yolo_overlay",
        help="Disable YOLO bounding box overlay",
    )
    parser.set_defaults(yolo_overlay=True)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load zone config if --zone is provided
    # ------------------------------------------------------------------
    if args.zone is not None:
        try:
            raw = json.load(open("config/zones.json"))
        except FileNotFoundError:
            print("ERROR: config/zones.json not found. Run from the project root.")
            sys.exit(1)

        zones = raw.get("zones", raw)
        if args.zone not in zones:
            available = ", ".join(zones.keys())
            print(f"ERROR: Zone '{args.zone}' not found. Available: {available}")
            sys.exit(1)

        zcfg = zones[args.zone]

        # Override video source from zone config
        if zcfg.get("video_source"):
            args.video = zcfg["video_source"]

        # Override mode from zone type
        args.mode = zcfg["type"]

        # Override area
        if zcfg.get("area_sqm") is not None:
            args.area_sqm = float(zcfg["area_sqm"])

        # Override ROI
        if zcfg.get("roi"):
            args.roi = json.dumps(zcfg["roi"])

        # General-zone specifics
        if zcfg.get("calibration_factor") is not None:
            args.calibration_factor = float(zcfg["calibration_factor"])

        # Gate-zone specifics
        if zcfg.get("tripwire") is not None:
            tw = zcfg["tripwire"]
            if isinstance(tw, dict):
                # Config format: {"p1": [x,y], "p2": [x,y]}
                args.tripwire = json.dumps([tw["p1"], tw["p2"]])
            else:
                args.tripwire = json.dumps(tw)

        if zcfg.get("in_direction") is not None:
            args.in_direction = zcfg["in_direction"]

    # ------------------------------------------------------------------
    # Parse JSON strings into Python objects and attach to args
    # ------------------------------------------------------------------
    args.roi_parsed = None
    if args.roi is not None:
        try:
            args.roi_parsed = json.loads(args.roi)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --roi is not valid JSON: {exc}")
            sys.exit(1)

    args.tripwire_parsed = None
    if args.tripwire is not None:
        try:
            args.tripwire_parsed = json.loads(args.tripwire)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --tripwire is not valid JSON: {exc}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if args.video is None:
        print("ERROR: Provide a video path or use --zone to auto-load one.")
        parser.print_usage()
        sys.exit(1)

    if args.mode == "gate" and args.tripwire_parsed is None:
        print("ERROR: --mode gate requires --tripwire (or use --zone).")
        sys.exit(1)

    return args


# ---------------------------------------------------------------------------
# Main playground class
# ---------------------------------------------------------------------------

class CVPlayground:
    """
    Isolated CV testing environment. Uses the same vision components
    as the production pipeline but with zero backend dependencies.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.frame_count = 0
        self._last_rendered: np.ndarray | None = None

        # Initialize YOLO (same model as production)
        from ultralytics import YOLO
        self.yolo = YOLO("yolov8n.pt")

        # Initialize EMA (same params as production)
        from core.smoothing import EMACounter
        self.ema = EMACounter(alpha=0.3, spike_threshold=2.0)

        if args.mode == "general":
            from vision.density_estimator import DenseEstimator
            from vision.flow_analyzer import OpticalFlowAnalyzer
            from vision.cross_calibrator import CrossCalibrator

            cal_factor = args.calibration_factor if args.calibration_factor is not None else 0.0025
            self.dense_estimator = DenseEstimator(calibration_factor=cal_factor)
            self.cross_calibrator = CrossCalibrator(initial_factor=cal_factor)

            # ROI may be None here if not provided — updated in run() after video opens
            roi = args.roi_parsed
            if roi is not None:
                self.roi: dict = roi
                zone_roi = (roi["x"], roi["y"], roi["w"], roi["h"])
            else:
                self.roi = {"x": 0, "y": 0, "w": 640, "h": 480}  # placeholder
                zone_roi = (0, 0, 640, 480)

            self.flow_analyzer = OpticalFlowAnalyzer(
                zone_rois={"test_zone": zone_roi}
            )

        elif args.mode == "gate":
            from vision.gate_counter import GateTripwireCounter
            tripwire = args.tripwire_parsed
            self.gate_counter = GateTripwireCounter(
                tripwire=(tuple(tripwire[0]), tuple(tripwire[1])),
                in_direction=args.in_direction,
                confidence_threshold=args.conf,
            )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame_general(self, frame: np.ndarray) -> dict:
        """Process one frame in general (dense) mode. Returns metrics dict."""
        roi = self.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

        # 1. YOLO detection
        results = self.yolo(
            frame, conf=self.args.conf, classes=[0],
            verbose=False, imgsz=self.args.imgsz,
        )
        yolo_boxes = results[0].boxes
        yolo_count = len(yolo_boxes)

        # 2. Dense estimation (MOG2)
        dense_count = self.dense_estimator.estimate(frame, (x, y, w, h))
        fg_pixels = self.dense_estimator.get_last_fg_pixels()

        # 3. Cross-calibration
        cal_factor = self.cross_calibrator.update(yolo_count, fg_pixels)
        self.dense_estimator.calibration_factor = cal_factor

        # 4. Auto-switch decision (mirrors production logic)
        switch_threshold = 40
        fg_override_threshold = 25_000
        fg_override = fg_pixels >= fg_override_threshold
        use_dense = (yolo_count >= switch_threshold or fg_override) and dense_count >= 0
        if use_dense:
            chosen_count = dense_count
            mode_label = "DENSE (MOG2) [FG OVERRIDE]" if fg_override else "DENSE (MOG2)"
        else:
            chosen_count = yolo_count
            mode_label = "YOLO"

        # 5. EMA smoothing
        smoothed = self.ema.update(chosen_count)
        was_rejected = self.ema.last_was_rejected

        # 6. Optical flow
        flow_results = self.flow_analyzer.analyze(frame)
        flow = flow_results.get("test_zone")
        flow_dir = flow.direction_degrees if flow else 0.0
        flow_mag = flow.magnitude if flow else 0.0
        is_stagnant = flow.is_stagnant if flow else False

        # 7. Density
        density = smoothed / self.args.area_sqm

        return {
            "frame": self.frame_count,
            "yolo_count": yolo_count,
            "dense_count": dense_count,
            "fg_pixels": fg_pixels,
            "cal_factor": cal_factor,
            "cal_confidence": self.cross_calibrator.confidence,
            "chosen_count": chosen_count,
            "mode": mode_label,
            "smoothed": smoothed,
            "ema_rejected": was_rejected,
            "density": density,
            "flow_dir": flow_dir,
            "flow_mag": flow_mag,
            "is_stagnant": is_stagnant,
            "yolo_boxes": yolo_boxes,
        }

    def process_frame_gate(self, frame: np.ndarray) -> dict:
        """Process one frame in gate mode. Returns metrics dict."""
        gate_count = self.gate_counter.process_frame(frame)

        # Plain YOLO for instantaneous count comparison
        results = self.yolo(
            frame, conf=self.args.conf, classes=[0],
            verbose=False, imgsz=self.args.imgsz,
        )
        yolo_count = len(results[0].boxes)

        return {
            "frame": self.frame_count,
            "yolo_count": yolo_count,
            "gate_in": gate_count.gate_in,
            "gate_out": gate_count.gate_out,
            "active_tracks": gate_count.active_tracks,
            "net_occupancy": gate_count.gate_in - gate_count.gate_out,
            "yolo_boxes": results[0].boxes,
        }

    # ------------------------------------------------------------------
    # Visual debug panels — GENERAL mode
    # ------------------------------------------------------------------

    def draw_yolo_overlay(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        display = frame.copy()
        roi = self.roi
        # Cyan ROI rectangle
        cv2.rectangle(
            display,
            (roi["x"], roi["y"]),
            (roi["x"] + roi["w"], roi["y"] + roi["h"]),
            (255, 255, 0), 2,
        )
        # YOLO bounding boxes
        if self.args.yolo_overlay and metrics["yolo_boxes"] is not None:
            for box in metrics["yolo_boxes"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                )
        cv2.putText(
            display, f"YOLO: {metrics['yolo_count']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )
        return display

    def draw_mog2_overlay(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        roi = self.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        mask = self.dense_estimator.get_last_mask()
        if mask is not None:
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            display = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)

        cv2.putText(
            display, f"FG pixels: {metrics['fg_pixels']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            display, f"Dense count: {metrics['dense_count']}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            display, f"Cal factor: {metrics['cal_factor']:.6f}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        warmup_label = "WARMING UP" if metrics["dense_count"] == -1 else ""
        if warmup_label:
            cv2.putText(
                display, warmup_label,
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
            )
        return display

    def draw_flow_overlay(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        display = frame.copy()
        cx = display.shape[1] // 2
        cy = display.shape[0] // 2
        angle_rad = np.radians(metrics["flow_dir"])
        arrow_len = min(100, max(5, int(metrics["flow_mag"] * 30)))
        end_x = int(cx + arrow_len * np.cos(angle_rad))
        end_y = int(cy + arrow_len * np.sin(angle_rad))
        color = (0, 0, 255) if metrics.get("is_stagnant") else (0, 255, 0)
        cv2.arrowedLine(display, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)
        label = "STAGNANT" if metrics.get("is_stagnant") else f"{metrics['flow_mag']:.1f} px/f"
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(
            display, f"Dir: {metrics['flow_dir']:.0f} deg",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
        return display

    def draw_stats_panel(self, metrics: dict, panel_size: tuple[int, int] = (640, 480)) -> np.ndarray:
        panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
        rejected_str = "YES" if metrics["ema_rejected"] else "no"
        stagnant_str = "YES" if metrics.get("is_stagnant") else "no"
        lines = [
            f"Frame: {metrics['frame']}",
            "",
            "=== DETECTION ===",
            f"YOLO count:    {metrics['yolo_count']}",
            f"Dense count:   {metrics['dense_count']}",
            f"FG pixels:     {metrics['fg_pixels']}",
            f"Active mode:   {metrics['mode']}",
            "",
            "=== SMOOTHING ===",
            f"Chosen raw:    {metrics['chosen_count']}",
            f"EMA smoothed:  {metrics['smoothed']:.1f}",
            f"EMA rejected:  {rejected_str}",
            f"Density:       {metrics['density']:.4f} p/sqm",
            "",
            "=== CALIBRATION ===",
            f"Cal factor:    {metrics['cal_factor']:.6f}",
            f"Cal confidence:{metrics['cal_confidence']:.0%}",
            "",
            "=== FLOW ===",
            f"Direction:     {metrics['flow_dir']:.0f} deg",
            f"Magnitude:     {metrics['flow_mag']:.2f} px/f",
            f"Stagnant:      {stagnant_str}",
        ]
        y = 25
        for line in lines:
            if "YES" in line:
                color = (0, 0, 255)
            elif "===" in line:
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)
            cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 22
        return panel

    # ------------------------------------------------------------------
    # Visual debug panels — GATE mode
    # ------------------------------------------------------------------

    def draw_gate_overlay(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        display = frame.copy()

        # Draw YOLO boxes
        if self.args.yolo_overlay and metrics["yolo_boxes"] is not None:
            for box in metrics["yolo_boxes"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                )

        # Draw tripwire line (red)
        tw = self.gate_counter.tripwire
        cv2.line(display, tw[0], tw[1], (0, 0, 255), 2)

        # IN/OUT counts near the tripwire midpoint
        mx = (tw[0][0] + tw[1][0]) // 2
        my = (tw[0][1] + tw[1][1]) // 2
        cv2.putText(
            display, f"IN:  {metrics['gate_in']}",
            (mx + 10, my - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.putText(
            display, f"OUT: {metrics['gate_out']}",
            (mx + 10, my + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

        # Frame counter and tracks
        cv2.putText(
            display, f"YOLO: {metrics['yolo_count']}  Tracks: {metrics['active_tracks']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )
        return display

    def draw_gate_stats(self, metrics: dict, panel_size: tuple[int, int] = (640, 480)) -> np.ndarray:
        panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
        net = metrics["net_occupancy"]
        net_color_str = "+" if net >= 0 else ""
        lines = [
            f"Frame: {metrics['frame']}",
            "",
            "=== GATE COUNTS ===",
            f"IN  (cumulative): {metrics['gate_in']}",
            f"OUT (cumulative): {metrics['gate_out']}",
            f"Net occupancy:    {net_color_str}{net}",
            "",
            "=== TRACKING ===",
            f"Active tracks:    {metrics['active_tracks']}",
            f"YOLO detections:  {metrics['yolo_count']}",
            "",
            "=== TRIPWIRE ===",
            f"Direction IN:     {self.args.in_direction}",
        ]
        y = 25
        for line in lines:
            if "===" in line:
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)
            cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 22
        return panel

    # ------------------------------------------------------------------
    # Composite renderer
    # ------------------------------------------------------------------

    def render(self, frame: np.ndarray, metrics: dict) -> None:
        panel_w, panel_h = 640, 480
        if self.args.mode == "general":
            tl = cv2.resize(self.draw_yolo_overlay(frame, metrics), (panel_w, panel_h))
            tr = cv2.resize(self.draw_mog2_overlay(frame, metrics), (panel_w, panel_h))
            bl = cv2.resize(self.draw_flow_overlay(frame, metrics), (panel_w, panel_h))
            br = self.draw_stats_panel(metrics, (panel_w, panel_h))
            top = np.hstack([tl, tr])
            bottom = np.hstack([bl, br])
            combined = np.vstack([top, bottom])
        else:
            left = cv2.resize(self.draw_gate_overlay(frame, metrics), (panel_w, panel_h))
            right = self.draw_gate_stats(metrics, (panel_w, panel_h))
            combined = np.hstack([left, right])

        self._last_rendered = combined
        cv2.imshow("CrowdFlow CV Playground", combined)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened():
            print(f"ERROR: Cannot open '{self.args.video}'")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If no ROI was provided for general mode, default to full frame
        if self.args.mode == "general" and self.args.roi_parsed is None:
            self.roi = {"x": 0, "y": 0, "w": vid_w, "h": vid_h}
            self.flow_analyzer.zone_rois["test_zone"] = (0, 0, vid_w, vid_h)
            print(f"No ROI specified — defaulting to full frame ({vid_w}x{vid_h})")

        print(f"Video:      {self.args.video}")
        print(f"Resolution: {vid_w}x{vid_h}")
        print(f"FPS: {fps:.1f}, Total frames: {total_frames}")
        print(f"Mode: {self.args.mode.upper()}")
        print(f"YOLO conf={self.args.conf}, imgsz={self.args.imgsz}")
        if self.args.mode == "general":
            print(f"ROI: {self.roi}")
            cal = self.args.calibration_factor if self.args.calibration_factor is not None else 0.0025
            print(f"Cal factor: {cal}")
        elif self.args.mode == "gate":
            print(f"Tripwire: {self.gate_counter.tripwire}")
            print(f"IN direction: {self.args.in_direction}")
        print()
        if not self.args.headless:
            print("Controls: SPACE=pause/resume  Q=quit  S=save frame  N=next frame (when paused)")
        print("=" * 60)

        # CSV setup
        csv_file = None
        csv_writer = None
        if self.args.export_csv:
            import csv as _csv
            csv_file = open(self.args.export_csv, "w", newline="")
            if self.args.mode == "general":
                fieldnames = [
                    "frame", "yolo_count", "dense_count", "fg_pixels",
                    "cal_factor", "cal_confidence", "chosen_count", "mode",
                    "smoothed", "ema_rejected", "density", "flow_dir", "flow_mag", "is_stagnant",
                ]
            else:
                fieldnames = [
                    "frame", "yolo_count", "gate_in", "gate_out",
                    "active_tracks", "net_occupancy",
                ]
            csv_writer = _csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        # Detect whether OpenCV GUI is available
        gui_available = not self.args.headless
        if gui_available:
            try:
                # Attempt a no-op imshow to detect headless environments
                test = np.zeros((1, 1, 3), dtype=np.uint8)
                cv2.imshow("CrowdFlow CV Playground", test)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
            except cv2.error:
                print("WARNING: OpenCV GUI unavailable — falling back to headless mode.")
                gui_available = False

        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print(f"\nEnd of video at frame {self.frame_count}")
                    break
                self.frame_count += 1

            # Process frame
            if self.args.mode == "general":
                metrics = self.process_frame_general(frame)
            else:
                metrics = self.process_frame_gate(frame)

            # Output
            if not gui_available:
                # Headless: print every 10th frame
                if self.frame_count % 10 == 0:
                    if self.args.mode == "general":
                        print(
                            f"F{metrics['frame']:04d} | "
                            f"YOLO:{metrics['yolo_count']:3d} | "
                            f"Dense:{metrics['dense_count']:3d} | "
                            f"FG:{metrics['fg_pixels']:6d} | "
                            f"Mode:{metrics['mode']:10s} | "
                            f"EMA:{metrics['smoothed']:6.1f} | "
                            f"Density:{metrics['density']:.4f} | "
                            f"Flow:{metrics['flow_mag']:.1f}@{metrics['flow_dir']:.0f}\u00b0"
                            f"{' REJECTED' if metrics['ema_rejected'] else ''}"
                        )
                    else:
                        print(
                            f"F{metrics['frame']:04d} | "
                            f"YOLO:{metrics['yolo_count']:3d} | "
                            f"IN:{metrics['gate_in']:3d} | "
                            f"OUT:{metrics['gate_out']:3d} | "
                            f"Net:{metrics['net_occupancy']:+3d} | "
                            f"Tracks:{metrics['active_tracks']:2d}"
                        )
            else:
                self.render(frame, metrics)

                wait_time = 0 if (paused or self.args.slow) else 1
                key = cv2.waitKey(wait_time) & 0xFF

                if key == ord('q'):
                    print(f"\nQuit at frame {self.frame_count}")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'PAUSED' if paused else 'RESUMED'} at frame {self.frame_count}")
                elif key == ord('s'):
                    if self._last_rendered is not None:
                        path = f"cv_debug_frame_{self.frame_count:04d}.jpg"
                        cv2.imwrite(path, self._last_rendered)
                        print(f"Saved debug frame to {path}")
                elif key == ord('n') and paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video")
                        break
                    self.frame_count += 1

            # CSV export (only when not paused to avoid duplicate rows)
            if csv_writer and not paused:
                export = {k: v for k, v in metrics.items() if k != "yolo_boxes"}
                # Keep only the declared fieldnames
                row = {k: export[k] for k in csv_writer.fieldnames if k in export}
                csv_writer.writerow(row)

        cap.release()
        if gui_available:
            cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
            print(f"Metrics exported to {self.args.export_csv}")

        # Summary
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {self.frame_count} frames processed")
        if self.args.mode == "general":
            ema_val = self.ema._ema if self.ema._ema is not None else 0.0
            print(f"Final EMA count:  {ema_val:.1f}")
            print(f"Final cal factor: {self.cross_calibrator.current_factor:.6f}")
            print(f"Cal confidence:   {self.cross_calibrator.confidence:.0%}")
            print(f"Cal samples:      {self.cross_calibrator.samples}")
        elif self.args.mode == "gate":
            print(f"Gate IN:          {self.gate_counter._cumulative_in}")
            print(f"Gate OUT:         {self.gate_counter._cumulative_out}")
            net = self.gate_counter._cumulative_in - self.gate_counter._cumulative_out
            print(f"Net occupancy:    {net:+d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    playground = CVPlayground(args)
    playground.run()


if __name__ == "__main__":
    main()
