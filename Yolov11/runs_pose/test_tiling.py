# -*- coding: utf-8 -*-
"""对比整图推理 vs 上下切片推理的 tip 误差，验证瓶颈是不是分辨率"""
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

ROOT = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521")
m = YOLO(r"D:\yolov11\runs_pose\ice80_yolo26\weights\best.pt")


def load_gt(lbl, W, H):
    B, K = [], []
    for line in open(lbl, encoding="utf-8"):
        f = line.split()
        if len(f) < 14:
            continue
        cx, cy, bw, bh = map(float, f[1:5])
        B.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        K.append([(float(f[5 + 3 * k]) * W, float(f[6 + 3 * k]) * H) for k in range(3)])
    return B, K


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def tiled(im, ntile=2, ov=0.12, sz=1280):
    W, H = im.size
    step = H // ntile
    boxes, kps = [], []
    for t in range(ntile):
        y0 = max(0, t * step - int(ov * step))
        y1 = min(H, (t + 1) * step + int(ov * step))
        arr = np.array(im.crop((0, y0, W, y1)))
        r = m.predict(source=arr, imgsz=sz, conf=0.25, verbose=False)[0]
        for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy()):
            cy = (b[1] + b[3]) / 2
            if cy < 8 or cy > (y1 - y0) - 8:
                continue
            boxes.append([b[0], b[1] + y0, b[2], b[3] + y0])
            kps.append([[p[0], p[1] + y0] for p in k])
    return boxes, kps


def evaluate(preds):
    errs, ious = [], []
    for ip in sorted((ROOT / "images" / "test").glob("*.jpg")):
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        B, K = load_gt(ROOT / "labels" / "test" / (ip.stem + ".txt"), W, H)
        pb, pk = preds(im, ip)
        cand = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(B) for pi, p in enumerate(pb)), reverse=True)
        ug, up = set(), set()
        for v, gi, pi in cand:
            if v < 0.3 or gi in ug or pi in up:
                continue
            ug.add(gi); up.add(pi)
            errs.append(abs(pk[pi][0][1] - K[gi][0][1]))
            ious.append(v)
    return np.array(errs), np.array(ious)


full = lambda im, ip: (m.predict(source=str(ip), imgsz=1536, conf=0.25, verbose=False)[0].boxes.xyxy.cpu().numpy(),
                       m.predict(source=str(ip), imgsz=1536, conf=0.25, verbose=False)[0].keypoints.xy.cpu().numpy())

for tag, fn in (("整图 imgsz1536", full),
                ("上下两片 imgsz1280", lambda im, ip: tiled(im, 2, 0.12, 1280)),
                ("上下两片 imgsz1536", lambda im, ip: tiled(im, 2, 0.12, 1536))):
    e, i = evaluate(fn)
    print("%-20s n=%2d tip误差 中位%4.0fpx 平均%4.0fpx | IoU中位%.2f" % (tag, len(e), np.median(e), e.mean(), np.median(i)))
