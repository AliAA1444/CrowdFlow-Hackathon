# CrowdFlow — Test Video Files

Place crowd video clips in this directory so the camera inference engine can
process them. Each file should be 30–120 seconds long at any resolution
(YOLOv8 handles internal resizing).

## Expected filenames

| Camera ID            | File                    | Zone                  |
|----------------------|-------------------------|-----------------------|
| cam_north_gate       | north_gate.mp4          | zone_north_gate       |
| cam_south_gate       | south_gate.mp4          | zone_south_gate       |
| cam_east_stand       | east_stand.mp4          | zone_east_stand       |
| cam_west_stand       | west_stand.mp4          | zone_west_stand       |
| cam_north_concourse  | north_concourse.mp4     | zone_north_concourse  |
| cam_south_concourse  | south_concourse.mp4     | zone_south_concourse  |
| cam_pitch_view       | pitch_view.mp4          | zone_pitch_view       |
| cam_activation_1     | activation_1.mp4        | zone_activation_1     |
| cam_activation_2     | activation_2.mp4        | zone_activation_2     |
| cam_parking          | parking.mp4             | zone_north_parking    |

## Where to get test videos

### Option 1 — VIRAT Dataset (recommended for realism)
- https://viratdata.org/
- Download ground-camera clips (VIRAT Ground Dataset)
- Contains parking lots, plazas, and pedestrian areas — ideal for density testing

### Option 2 — MOT16 / MOT20 Benchmark
- https://motchallenge.net/
- MOT20 specifically contains high-density crowd sequences
- Download sequences, extract frames, reassemble as .mp4 with ffmpeg:
  ```
  ffmpeg -r 25 -i %06d.jpg -vcodec libx264 output.mp4
  ```

### Option 3 — YouTube / Creative Commons clips
Search for:
- "stadium crowd footage" → filter by CC licence
- "CCTV crowd footage pedestrian"
- "football stadium turnstile camera"

Download with yt-dlp:
```bash
pip install yt-dlp
yt-dlp -f "best[ext=mp4]" -o "north_gate.mp4" "<URL>"
```

### Option 4 — Synthetic test clip (zero setup)
Generate a short clip of walking people with any crowd simulation tool,
or use your laptop webcam as a test source by setting `video_file` to `0`
in CAMERA_FEEDS (integer device index — change `cv2.VideoCapture(video_file)`
to handle both int and str paths).

## Notes
- Missing files are handled gracefully: `inference/fallback_simulator.py`
  automatically covers any zone whose video file is absent.
- Videos loop continuously — no minimum length required.
- Any resolution works; 480p or 720p is sufficient and faster to process.
