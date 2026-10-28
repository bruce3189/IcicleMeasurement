"""YOLO11 pose 训练（冰凌 80 张）—— 对照组

环境：D:\318\envs\yolov11\python.exe （ultralytics 8.3.151 / torch 2.0.0+cpu）
用法：D:\318\envs\yolov11\python.exe D:\yolov11\runs_pose\train_pose11.py
"""
from pathlib import Path

from ultralytics import YOLO

DATA = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521\dataset.yaml")
WEIGHT = Path(r"D:\yolov11\weights\yolo11n-pose.pt")
PROJECT = Path(r"D:\yolov11\runs_pose")
NAME = "ice80_yolo11"

IMGSZ = 1280  # 长边；配 rect=True 按原比例成批（实际约 548x1280）

if __name__ == "__main__":
    model = YOLO(str(WEIGHT))
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=IMGSZ,
        rect=True,
        batch=2,
        workers=0,
        device="cpu",
        cache=False,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        seed=42,
        deterministic=True,
        patience=40,
        plots=True,
        val=True,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.3,
    )
    print("TRAIN_DONE", PROJECT / NAME / "weights" / "best.pt")
