import csv
from pathlib import Path
import cv2

source = Path(r'C:\Users\Administrator\Desktop\images')
root = Path(r'C:\Users\Administrator\Documents\Codex\2026-10-24\continuing-from-research-the-user-s\outputs\ice_pose_runs\predict_images_400_202610242')
labels = root / 'labels'
out_dir = root / 'thin_overlay'
out_dir.mkdir(parents=True, exist_ok=True)
summary = []

for image_path in sorted(source.glob('*.jpg')):
    image = cv2.imread(str(image_path))
    if image is None:
        continue
    h, w = image.shape[:2]
    thickness = max(1, round(min(w, h) / 1000))
    point_radius = max(2, round(min(w, h) / 550))
    label_path = labels / (image_path.stem + '.txt')
    count = 0
    if label_path.exists():
        for line in label_path.read_text(encoding='utf-8').splitlines():
            vals = [float(x) for x in line.split()]
            if len(vals) < 12:
                continue
            count += 1
            cls, cx, cy, bw, bh = vals[:5]
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            box_conf = vals[11]
            color = (255, 90, 20)  # slim blue-orange box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            # The dataset order is tip first, base second.
            tip = (int(vals[5] * w), int(vals[6] * h))
            base = (int(vals[8] * w), int(vals[9] * h))
            cv2.line(image, tip, base, (40, 180, 255), thickness, cv2.LINE_AA)
            cv2.circle(image, tip, point_radius, (0, 70, 255), -1, cv2.LINE_AA)
            cv2.circle(image, base, point_radius, (0, 220, 255), -1, cv2.LINE_AA)
            text = f'{box_conf:.2f}'
            font_scale = 0.42
            text_y = max(18, y1 - 4)
            cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 235, 120), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / image_path.name), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    summary.append({'image': image_path.name, 'detections': count})

with (root / 'image_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['image', 'detections'])
    writer.writeheader(); writer.writerows(summary)
print(f'REDRAWN={len(summary)} OUTPUT={out_dir}')
