"""YOLO26 pose 训练（冰凌 80 张）

环境：D:\318\envs\yolo26\python.exe   （ultralytics 8.4.183 / torch 2.8.0+cpu）
用法：把本文件放在不含 ultralytics 源码目录的位置运行，
      例如  D:\318\envs\yolo26\python.exe D:\yolov11\runs_pose\train_pose26.py
"""
from pathlib import Path

from ultralytics import YOLO

DATA = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521\dataset.yaml")
WEIGHT = Path(r"D:\yolov11\weights\yolo26n-pose.pt")
PROJECT = Path(r"D:\yolov11\runs_pose")
NAME = "ice80_yolo26"

# 原图 1644x3840（细长竖图）。train/val 的 imgsz 只能传整数，
# 配 rect=True 后按原比例成批（长边=1280，实际约 548x1280），避免方形填充浪费一半算力
IMGSZ = 1280

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
        # 翻转增强对“左右角点”有语义风险，横向翻转会把 base_left/right 互换，关掉
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,      # 冰凌是细长目标，mosaic 拼接容易造出断头样本，初级模型先关
        degrees=0.0,
        translate=0.05,
        scale=0.3,
    )
    print("TRAIN_DONE", PROJECT / NAME / "weights" / "best.pt")
