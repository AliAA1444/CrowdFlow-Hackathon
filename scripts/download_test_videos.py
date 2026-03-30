#!/usr/bin/env python3
"""CrowdFlow — Test video downloader / synthetic generator.

Downloads Creative Commons crowd footage and saves it to data/videos/ with
the filenames expected by CAMERA_FEEDS in config.py.

If any download fails (network error, URL expired, etc.), a synthetic test
video is generated instead using OpenCV.  The synthetic clips contain
animated "person blobs" moving across the frame, which gives YOLO something
plausible to process and ensures the demo always works offline.

Usage:
    python scripts/download_test_videos.py           # download + fill gaps
    python scripts/download_test_videos.py --synth   # generate all synthetically
    python scripts/download_test_videos.py --list    # show expected filenames

Recommended manual sources (if you want real footage):
  • VIRAT Dataset:   https://viratdata.org/
  • MOT20 Benchmark: https://motchallenge.net/data/MOT20/
  • Pixabay Video:   https://pixabay.com/videos/search/crowd/  (free, no login)
  • Pexels Video:    https://www.pexels.com/search/videos/crowd/  (free API key)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CAMERA_FEEDS

VIDEOS_DIR = ROOT / "data" / "videos"

# ── Public-domain / CC video candidates ──────────────────────────────────────
# These are Wikimedia Commons direct-download MP4 URLs (CC-BY or CC0).
# The script tries them in order and falls back to synthetic if they fail.
# URL→target_filename mapping (a single source can seed multiple filenames).
DOWNLOAD_CANDIDATES: list[tuple[str, list[str]]] = [
    # Crowd walking through a corridor – seeds concourse and gate clips
    (
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/"
        "People_Walking_In_Busy_Downtown_Corridor.webm/"
        "People_Walking_In_Busy_Downtown_Corridor.webm.480p.vp9.webm",
        ["north_concourse.mp4", "south_concourse.mp4", "north_gate.mp4", "south_gate.mp4"],
    ),
    # Busy street / pedestrian crowd – seeds stands and activation zones
    (
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/"
        "Crowd_at_the_2010_FIFA_World_Cup_Vuvuzelas.ogv/"
        "Crowd_at_the_2010_FIFA_World_Cup_Vuvuzelas.ogv.480p.vp9.webm",
        ["east_stand.mp4", "west_stand.mp4", "activation_1.mp4", "activation_2.mp4"],
    ),
    # Open-area crowd – seeds pitch view and parking
    (
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/7/74/"
        "Crowd_gathering_-_Wroclaw_-_Poland_-_panoramic_%2814767460215%29.webm/"
        "Crowd_gathering_-_Wroclaw_-_Poland_-_panoramic_%2814767460215%29.webm.480p.vp9.webm",
        ["pitch_view.mp4", "parking.mp4"],
    ),
]

# All filenames the system expects
ALL_EXPECTED = {
    cam: Path(cfg["video_file"]).name
    for cam, cfg in CAMERA_FEEDS.items()
}


# ── Download helpers ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CrowdFlow/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            chunk = 65_536
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    fh.write(data)
                    written += len(data)
                    if total:
                        pct = written / total * 100
                        print(f"\r  {dest.name}: {pct:5.1f}%", end="", flush=True)
            tmp.rename(dest)
        print(f"\r  ✓ {dest.name} ({written // 1024} KB)")
        return True
    except (urllib.error.URLError, OSError, Exception) as exc:
        print(f"\r  ✗ {dest.name}: download failed ({exc.__class__.__name__})")
        if dest.with_suffix(".tmp").exists():
            dest.with_suffix(".tmp").unlink(missing_ok=True)
        return False


def _copy_as(src: Path, dest: Path) -> bool:
    """Copy *src* to *dest* (for seeding multiple filenames from one download)."""
    import shutil
    try:
        shutil.copy2(src, dest)
        print(f"  ✓ {dest.name} (copied from {src.name})")
        return True
    except OSError as exc:
        print(f"  ✗ {dest.name}: copy failed ({exc})")
        return False


# ── Synthetic video generator ─────────────────────────────────────────────────

def generate_synthetic_video(
    dest: Path,
    *,
    duration_s: int = 30,
    fps: int = 15,
    width: int = 640,
    height: int = 480,
    n_blobs: int | None = None,
) -> None:
    """
    Generate a synthetic crowd video with OpenCV.

    Draws N animated ellipse blobs (person proxies) moving in random
    directions across a dark background.  YOLO won't detect these as
    real people, but the video gives the camera engine valid frames to
    process and lets the fallback simulator stay active for the demo.

    For a real demo, replace these with actual crowd footage — see the
    README in data/videos/ for guidance.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print(f"  ✗ {dest.name}: opencv-python not installed — skipping synthetic gen")
        return

    if n_blobs is None:
        # Vary blob count by filename to simulate different density zones
        name = dest.stem
        if "stand" in name or "gate" in name:
            n_blobs = random.randint(18, 30)
        elif "concourse" in name:
            n_blobs = random.randint(10, 20)
        elif "parking" in name or "activation" in name:
            n_blobs = random.randint(4, 12)
        else:
            n_blobs = random.randint(8, 16)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))

    # Initialise blob positions, velocities, and colours
    rng = random.Random(abs(hash(dest.stem)))
    blobs = [
        {
            "x": rng.uniform(40, width  - 40),
            "y": rng.uniform(40, height - 40),
            "vx": rng.uniform(-1.8, 1.8),
            "vy": rng.uniform(-1.8, 1.8),
            "rx": rng.randint(10, 18),
            "ry": rng.randint(14, 22),
            "color": (rng.randint(180, 255), rng.randint(100, 200), rng.randint(80, 180)),
            "angle": rng.uniform(0, 360),
        }
        for _ in range(n_blobs)
    ]

    total_frames = duration_s * fps
    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Faint grid lines for visual depth
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 0), (gx, height), (20, 20, 20), 1)
        for gy in range(0, height, 60):
            cv2.line(frame, (0, gy), (width, gy), (20, 20, 20), 1)

        for b in blobs:
            # Move blob
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            b["angle"] += 1.2

            # Bounce off walls
            if b["x"] - b["rx"] < 0:   b["x"] = b["rx"];         b["vx"] = abs(b["vx"])
            if b["x"] + b["rx"] > width:  b["x"] = width - b["rx"]; b["vx"] = -abs(b["vx"])
            if b["y"] - b["ry"] < 0:   b["y"] = b["ry"];          b["vy"] = abs(b["vy"])
            if b["y"] + b["ry"] > height: b["y"] = height - b["ry"];b["vy"] = -abs(b["vy"])

            # Slight random wobble
            b["vx"] += rng.uniform(-0.1, 0.1)
            b["vy"] += rng.uniform(-0.1, 0.1)
            b["vx"] = max(-2.5, min(2.5, b["vx"]))
            b["vy"] = max(-2.5, min(2.5, b["vy"]))

            cx, cy = int(b["x"]), int(b["y"])
            # Shadow
            cv2.ellipse(frame, (cx + 2, cy + 3), (b["rx"], b["ry"]),
                        b["angle"], 0, 360, (10, 10, 10), -1)
            # Body
            cv2.ellipse(frame, (cx, cy), (b["rx"], b["ry"]),
                        b["angle"], 0, 360, b["color"], -1)
            # Head highlight
            hx = int(cx + b["rx"] * 0.3 * math.cos(math.radians(b["angle"] - 30)))
            hy = int(cy + b["ry"] * 0.3 * math.sin(math.radians(b["angle"] - 30)))
            cv2.circle(frame, (hx, hy), max(3, b["rx"] // 3),
                       tuple(min(255, c + 60) for c in b["color"]), -1)

        # Timestamp watermark
        cv2.putText(frame, f"SYNTHETIC  frame {frame_idx:04d}",
                    (8, height - 8), cv2.FONT_HERSHEY_PLAIN, 0.85, (60, 60, 60), 1)

        writer.write(frame)

    writer.release()
    size_kb = dest.stat().st_size // 1024
    print(f"  ✓ {dest.name} (synthetic, {n_blobs} blobs, {duration_s}s, {size_kb} KB)")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(force_synthetic: bool = False) -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nCrowdFlow — test video setup  ({VIDEOS_DIR})\n")

    needed: set[str] = set()
    for cam_id, filename in ALL_EXPECTED.items():
        dest = VIDEOS_DIR / filename
        if not dest.exists():
            needed.add(filename)
        else:
            print(f"  ✓ {filename} already present")

    if not needed:
        print("\n  All video files already present — nothing to do.\n")
        return

    print(f"\n  {len(needed)} file(s) needed: {', '.join(sorted(needed))}\n")

    if force_synthetic:
        print("  [--synth] Skipping downloads, generating all synthetically.\n")
        for filename in sorted(needed):
            generate_synthetic_video(VIDEOS_DIR / filename)
        _final_report()
        return

    # ── Try downloads first ───────────────────────────────────────────────────
    downloaded_sources: dict[str, Path] = {}   # url → local path of first success

    for url, target_filenames in DOWNLOAD_CANDIDATES:
        # Pick targets from this source that are still needed
        targets_needed = [fn for fn in target_filenames if fn in needed]
        if not targets_needed:
            continue

        primary_dest = VIDEOS_DIR / targets_needed[0]
        print(f"  Trying: {url[:72]}...")

        if _download(url, primary_dest):
            downloaded_sources[url] = primary_dest
            needed.discard(targets_needed[0])

            # Copy to additional targets from same source
            for fn in targets_needed[1:]:
                dest = VIDEOS_DIR / fn
                if fn in needed and _copy_as(primary_dest, dest):
                    needed.discard(fn)
        else:
            # Download failed — generate synthetically for all targets from this source
            for fn in targets_needed:
                if fn in needed:
                    generate_synthetic_video(VIDEOS_DIR / fn)
                    needed.discard(fn)

    # ── Synthetic fallback for anything still missing ─────────────────────────
    if needed:
        print(f"\n  Generating {len(needed)} remaining file(s) synthetically...")
        for filename in sorted(needed):
            generate_synthetic_video(VIDEOS_DIR / filename)

    _final_report()


def _final_report() -> None:
    print("\n  ── Final status ──")
    all_ok = True
    for cam_id, filename in sorted(ALL_EXPECTED.items()):
        dest = VIDEOS_DIR / filename
        if dest.exists():
            size_kb = dest.stat().st_size // 1024
            print(f"  ✓  {filename:<30}  ({size_kb:>6} KB)  [{cam_id}]")
        else:
            print(f"  ✗  {filename:<30}  MISSING            [{cam_id}]")
            all_ok = False

    status = "All files ready" if all_ok else "Some files still missing"
    print(f"\n  {status}. Run `python run_all.py` to start CrowdFlow.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth", action="store_true", help="Generate all videos synthetically (skip downloads)")
    parser.add_argument("--list",  action="store_true", help="List expected filenames and exit")
    args = parser.parse_args()

    if args.list:
        print("\nExpected video files (data/videos/):\n")
        for cam_id, filename in sorted(ALL_EXPECTED.items()):
            dest = VIDEOS_DIR / filename
            status = "✓ present" if dest.exists() else "✗ missing"
            print(f"  {status}  {filename:<30}  [{cam_id}]")
        print()
        return

    run(force_synthetic=args.synth)


if __name__ == "__main__":
    main()
