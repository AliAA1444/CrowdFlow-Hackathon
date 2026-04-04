import cv2
import numpy as np
import sys
import json

def calibrate(video_path: str, roi=None) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    frame_count = 0
    fg_pixel_history = []

    print("جاري تدريب نموذج الخلفية (أول 60 إطار)... يرجى الانتظار.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame

        fg_mask = bg_sub.apply(roi_frame, learningRate=0.005)
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_pixels = cv2.countNonZero(cleaned)

        frame_count += 1
        if frame_count <= 60: continue

        fg_pixel_history.append(fg_pixels)

        if frame_count % 30 == 0:
            display = np.hstack([cv2.resize(roi_frame, (480, 360)), cv2.resize(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR), (480, 360))])
            cv2.imshow("Original vs Mask", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    if not fg_pixel_history: return

    median_fg = np.median(fg_pixel_history)
    print(f"\n متوسط البكسلات المتحركة: {median_fg:.0f}")
    
    try:
        manual_count = int(input(" قدر عدد الأشخاص الذين رأيتهم في الشاشة: "))
    except:
        manual_count = 50

    cal_factor = manual_count / median_fg if median_fg > 0 else 0.001
    print(f"\n انسخ هذا السطر وضعه في ملف zones.json:")
    print(f' "calibration_factor": {cal_factor:.6f}\n')

if __name__ == "__main__":
    video = sys.argv[1]
    roi_arg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    calibrate(video, roi_arg)