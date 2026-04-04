import cv2
import os
import glob

# قائمة بالمقاطع التي نريد إنتاجها (المسار، والاسم الجديد)
videos_to_make = [
    {
        "folder": "/Users/ali/Documents/ClaudeCode/CrowdFlow/MOT20/train/MOT20-05/img1/",
        "output_name": "feeds/real_stands_dense.mp4" # لاحظ أزلنا النقطتين لكي يحفظ داخل المشروع
    },
    {
        "folder": "/Users/ali/Documents/ClaudeCode/CrowdFlow/MOT20/train/MOT20-03/img1/",
        "output_name": "feeds/real_concourse.mp4"
    }
]

for vid in videos_to_make:
    image_folder = vid["folder"]
    video_name = vid["output_name"]

    print(f"\nجاري البحث عن الصور في: {image_folder}...")
    images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

    if not images:
        print(f"تنبيه: لم يتم العثور على صور في هذا المسار. سنتجاوزه.")
        continue

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 25, (width, height))

    print(f"جاري تحويل {len(images)} صورة لإنتاج {video_name}... يرجى الانتظار.")

    for image in images:
        video.write(cv2.imread(image))

    video.release()
    print(f"✅ تم بنجاح! الفيديو موجود الآن في: {video_name}")

print("\n🎉 انتهت العملية بالكامل!")