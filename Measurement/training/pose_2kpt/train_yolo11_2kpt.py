from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import ultralytics.data.dataset as dataset_module

# The restricted Windows runtime denies multiprocessing Pipe creation. Ultralytics
# only needs an iterator here, so validate/cache labels sequentially.
class SequentialPool:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def imap(self, func, iterable):
        return map(func, iterable)

dataset_module.ThreadPool = SequentialPool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
model = YOLO(str(PROJECT_ROOT.parent / 'history_archive' / 'yolov11_legacy' / 'weights' / 'yolo11n-pose.pt'))
model.train(
    data=str(PROJECT_ROOT.parent / 'final_results' / 'training' / 'dataset' / 'dataset.yaml'),
    imgsz=1280,
    rect=True,
    epochs=100,
    patience=40,
    batch=2,
    device='cpu',
    workers=0,
    cache=False,
    project=str(PROJECT_ROOT / 'training' / 'pose_2kpt' / 'runs'),
    name='yolo11n_pose_78_2kpt_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
    exist_ok=False,
    seed=42,
    deterministic=True,
    plots=True,
    val=True,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.0,
    degrees=0.0,
    translate=0.05,
    scale=0.3,
)
