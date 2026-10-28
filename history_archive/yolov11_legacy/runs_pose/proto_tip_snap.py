# -*- coding: utf-8 -*-
"""原型：tip 吸附后处理。框内纹理能量向下延伸，找到冰凌真正的底端，替代模型回归的 tip。
只改 tip 的 y，x 保持模型输出（或取该行纹理最强列）。"""
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

ROOT = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521")
W_PT = r"D:\yolov11\runs_pose\ice80_yolo26\weights\best.pt"


def load_gt(lbl, W, H):
    boxes, kps = [], []
    for line in open(lbl, encoding="utf-8"):
        f = line.split()
        if len(f) < 14:
            continue
        cx, cy, bw, bh = map(float, f[1:5])
        kp = [(float(f[5 + 3 * k]) * W, float(f[6 + 3 * k]) * H) for k in range(3)]
        boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        kps.append(kp)
    return boxes, kps


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def row_energy(gray, x1, x2, y1, y2):
    """逐行纹理能量：内缩 25% 宽度，取 |上下差分| 的列均值"""
    W = gray.shape[1]
    xa = int(max(0, x1 + 0.25 * (x2 - x1)))
    xb = int(min(W, x2 - 0.25 * (x2 - x1)))
    if xb - xa < 4:
        xa, xb = int(x1), int(x2)
    y1 = max(1, int(y1)); y2 = min(gray.shape[0] - 1, int(y2))
    seg = gray[y1:y2, xa:xb].astype(np.float32)
    if seg.size == 0:
        return np.zeros(0), y1
    d = np.abs(np.diff(seg, axis=0))
    return d.mean(axis=1), y1


def snap_tip(gray, box, tip_y, k=0.35, extend=0.7):
    """从 box 顶部扫到 box 底部以下 extend*bh，返回最靠下的强纹理行"""
    x1, y1, x2, y2 = box
    bh = y2 - y1
    lo, hi = y1, y2 + extend * bh
    e, y0 = row_energy(gray, x1, x2, lo, hi)
    if e.size < 5:
        return tip_y
    inside = int(max(0, (y2 - y0) * 0.7))
    base = np.median(e[:max(5, inside)]) if inside > 5 else np.median(e)
    thr = max(k * base, 1.0)
    rows = np.where(e > thr)[0]
    if rows.size == 0:
        return tip_y
    # 取最靠下的行，但要连续（避免零星噪点）：要求该行附近 3 行里至少 2 行超过阈值
    rel = e > thr
    cand = 0
    for i in range(len(rel) - 3, -1, -1):
        if rel[i] and rel[i:i + 3].sum() >= 2:
            cand = i
            break
    return float(y0 + cand)


def main():
    model = YOLO(W_PT)
    before, after, heights = [], [], []
    for img_path in sorted((ROOT / "images" / "test").glob("*.jpg")):
        lbl = ROOT / "labels" / "test" / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gray = np.array(img.convert("L"), dtype=np.float32)
        gts, gk = load_gt(lbl, W, H)
        r = model.predict(source=str(img_path), imgsz=1536, conf=0.25, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy()
        pk = r.keypoints.xy.cpu().numpy()
        cand = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(pb)), reverse=True)
        ug, up = set(), set()
        for v, gi, pi in cand:
            if v < 0.3 or gi in ug or pi in up:
                continue
            ug.add(gi); up.add(pi)
            b = pb[pi]
            tip = snap_tip(gray, b, pk[pi][0][1])
            gt = gk[gi][0]
            h = gts[gi][3] - gts[gi][1]
            before.append(abs(pk[pi][0][1] - gt[1]))
            after.append(abs(tip - gt[1]))
            heights.append(h)
    b, a = np.array(before), np.array(after)
    print("tip 竖直误差（%d 个匹配框）" % len(b))
    print("  后处理前：中位 %.0fpx  平均 %.0fpx  >10%%框高 %d 个" % (
        np.median(b), b.mean(), int((b / np.array(heights) > 0.10).sum())))
    print("  吸附之后：中位 %.0fpx  平均 %.0fpx  >10%%框高 %d 个" % (
        np.median(a), a.mean(), int((a / np.array(heights) > 0.10).sum())))
    print("  改善 %d 个 / 变差 %d 个" % (int((a < b - 5).sum()), int((a > b + 5).sum())))


if __name__ == "__main__":
    main()
