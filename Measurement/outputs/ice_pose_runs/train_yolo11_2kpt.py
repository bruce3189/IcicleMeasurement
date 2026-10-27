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

model = YOLO(r'D:\yolov11\weights\yolo11n-pose.pt')
model.train(
    data=r'C:\Users\Administrator\Documents\Codex\2026-10-24\continuing-from-research-the-user-s\outputs\ice_pose_runs\dataset_冰凌_两点_78_20261023\dataset.yaml',
    imgsz=1280,
    rect=True,
    epochs=100,
    patience=40,
    batch=2,
    device='cpu',
    workers=0,
    cache=False,
    project=r'C:\Users\Administrator\Documents\Codex\2026-10-24\continuing-from-research-the-user-s\outputs\ice_pose_runs',
    name='yolo11n_pose_78_2kpt',
    exist_ok=True,
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
