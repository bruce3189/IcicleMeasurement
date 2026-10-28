import csv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

project_root = Path(__file__).resolve().parents[2]
source = project_root / 'assets' / 'rgb'
weight = project_root.parent / 'final_results' / 'model' / 'best.pt'
project = project_root / 'outputs' / 'ice_pose_runs'
name = 'predict_2kpt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
model = YOLO(str(weight))
summary_path = project / name / 'image_summary.csv'
points_path = project / name / 'keypoints.csv'
summary_path.parent.mkdir(parents=True, exist_ok=False)
summary_fields = ['image', 'detections', 'prediction_image']
point_fields = ['image', 'detection_index', 'box_confidence', 'apex_x_px', 'apex_y_px', 'apex_confidence', 'base_x_px', 'base_y_px', 'base_confidence']

with summary_path.open('w', newline='', encoding='utf-8-sig') as sf, points_path.open('w', newline='', encoding='utf-8-sig') as pf:
    sw = csv.DictWriter(sf, fieldnames=summary_fields)
    pw = csv.DictWriter(pf, fieldnames=point_fields)
    sw.writeheader()
    pw.writeheader()
    count = 0
    total_detections = 0
    # This directory was exclusively created above; reuse it for images and CSVs.
    for result in model.predict(source=str(source), imgsz=1280, rect=True, batch=1, device='cpu', conf=0.25, save=True, save_txt=True, save_conf=True, project=str(project), name=name, exist_ok=True, stream=True, verbose=False):
        count += 1
        n = len(result.boxes) if result.boxes is not None else 0
        total_detections += n
        sw.writerow({'image': result.path and Path(result.path).name, 'detections': n, 'prediction_image': str(Path(result.save_dir) / Path(result.path).name)})
        if result.keypoints is not None:
            kps = result.keypoints.data.cpu().numpy()
            box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            for j, points in enumerate(kps):
                row = {'image': Path(result.path).name, 'detection_index': j, 'box_confidence': float(box_conf[j]) if len(box_conf) > j else ''}
                if len(points) > 0:
                    row.update({'apex_x_px': float(points[0][0]), 'apex_y_px': float(points[0][1]), 'apex_confidence': float(points[0][2]) if len(points[0]) > 2 else ''})
                if len(points) > 1:
                    row.update({'base_x_px': float(points[1][0]), 'base_y_px': float(points[1][1]), 'base_confidence': float(points[1][2]) if len(points[1]) > 2 else ''})
                pw.writerow(row)
        if count % 25 == 0:
            sf.flush(); pf.flush()
            print(f'PROGRESS {count}/400 images, {total_detections} detections', flush=True)
    sf.flush(); pf.flush()
print(f'PREDICTION_DONE images={count} detections={total_detections}', flush=True)
print(f'OUTPUT={summary_path.parent}', flush=True)
