from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
from ultralytics import YOLO


model = YOLO(str(PROJECT_ROOT / 'yolo11n.pt'))
model.train(
    # 用相对名：ultralytics 会在 ultralytics/cfg/datasets/ 下递归查找
    # 不要改成 PROJECT_ROOT / 'iceice.yaml'——该文件不存在（2026-10-26 迁移时踩过）
    data='iceice.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    cache=False,
    workers=0,
    conf=0.626,
    iou=0.65,
    max_det=300,
)