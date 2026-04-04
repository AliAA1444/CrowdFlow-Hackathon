import cv2
import sys

def select_roi(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return

    h, w = frame.shape[:2]
    print(f"Video resolution: {w}x{h}")
    print("ارسم مستطيلاً بالماوس على الشاشة. اضغط ENTER للتأكيد، أو C للإلغاء.")

    roi = cv2.selectROI("Select ROI - ENTER to confirm", frame, fromCenter=False)
    cv2.destroyAllWindows()
    cap.release()

    x, y, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    if rw == 0 or rh == 0:
        print("لم يتم تحديد منطقة.")
        return

    print(f"\n{'='*50}")
    print(f" انسخ هذا السطر وضعه في ملف zones.json:")
    print(f' "roi": {{"x": {x}, "y": {y}, "w": {rw}, "h": {rh}}}')
    print(f"\n{'='*50}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("الاستخدام: python tools/roi_selector.py <video_path>")
        sys.exit(1)
    select_roi(sys.argv[1])