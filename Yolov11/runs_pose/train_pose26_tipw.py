"""YOLO26 pose 训练（冰凌 80 张）—— tip 加权版

在 train_pose26.py 基础上，把 tip（第 0 个关键点）的损失权重放大 TIP_W 倍：
  1) RLE 损失（yolo26 的关键点回归主损失）：PoseLoss26.target_weights[0] *= TIP_W
  2) pose 损失（KeypointLoss）：按关键点乘 [TIP_W, 1, 1]
两处都只影响训练梯度，不碰验证指标的 OKS sigmas，因此 mAP 与上一轮可直接对比。

环境：D:\318\envs\yolo26\python.exe   （ultralytics 8.4.183 / torch 2.8.0+cpu）
用法：cd /d/yolov11/runs_pose && python train_pose26_tipw.py
"""
from pathlib import Path

import torch

import ultralytics.utils.loss as L
from ultralytics import YOLO

TIP_W = 3.0  # tip 权重倍数，base 两个点保持 1.0

DATA = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521\dataset.yaml")
WEIGHT = Path(r"D:\yolov11\weights\yolo26n-pose.pt")
PROJECT = Path(r"D:\yolov11\runs_pose")
NAME = "ice80_yolo26_tipw"

IMGSZ = 1280  # rect=True，长边 1280

# ---------- 损失加权补丁 ----------
_orig_p26_init = L.PoseLoss26.__init__


def _p26_init(self, model, *a, **k):
    _orig_p26_init(self, model, *a, **k)
    if self.rle_loss is not None:
        w = self.target_weights.clone()
        w[0] = float(w[0]) * TIP_W
        self.target_weights = w.to(self.device)
        print(f"[tip-weight] RLE target_weights -> {self.target_weights.tolist()}")


L.PoseLoss26.__init__ = _p26_init

_orig_kp_fwd = L.KeypointLoss.forward


def _kp_fwd(self, pred_kpts, gt_kpts, kpt_mask, area):
    d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
    klf = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
    e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
    w = torch.tensor([TIP_W, 1.0, 1.0], device=e.device).view(1, -1)
    return (klf.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask * w)).mean()


L.KeypointLoss.forward = _kp_fwd
# --------------------------------

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
