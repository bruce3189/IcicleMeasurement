from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
from ultralytics import YOLO
import os

# 方法1：下载并使用COCO8示例数据集
model = YOLO(str(PROJECT_ROOT / 'yolo11n.pt'))
model.train(
    data='coco8.yaml',  # 这会自动下载COCO8数据集
    epochs=30,
    imgsz=640,
    batch=-1,
    cache=False,
    workers=0,

)