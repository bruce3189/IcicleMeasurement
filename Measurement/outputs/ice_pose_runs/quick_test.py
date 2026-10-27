from pathlib import Path
from ultralytics import YOLO
root = Path(r'C:\Users\Administrator\Documents\Codex\2026-10-24\continuing-from-research-the-user-s\outputs\ice_pose_runs')
test_dir = root / 'dataset_冰凌_两点_78_20261023' / 'images' / 'test'
image = sorted(test_dir.glob('*.jpg'))[0]
model = YOLO(str(root / 'yolo11n_pose_78_2kpt' / 'quick_test_best_snapshot.pt'))
result = model.predict(source=str(image), imgsz=640, conf=0.25, device='cpu', save=True, project=str(root), name='quick_test', exist_ok=True, verbose=False)[0]
print('IMAGE', image)
print('OUTPUT', result.save_dir)
print('DETECTIONS', len(result.boxes))
if result.keypoints is not None:
    print('KEYPOINTS_XY_CONF', result.keypoints.data.cpu().numpy().tolist())
