from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
from ultralytics import YOLO

model = YOLO(str(PROJECT_ROOT / 'yolo11n.pt'))
print(model.task)
print(model.names)