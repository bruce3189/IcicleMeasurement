"""对比 YOLO11 / YOLO26 两个 pose 模型在 test 集上的表现

指标：
  1. 检出数 vs 真实框数（漏检/多检）
  2. 关键点像素误差（只统计 GT 中 v>0 的点），按框高归一化
  3. PCK@0.05 / PCK@0.10（误差 < 框高的 5% / 10% 记为命中）

用法：
  D:\318\envs\yolo26\python.exe D:\yolov11\runs_pose\compare_pose.py
  （YOLO11 的 best.pt 需用 8.3 环境读取，若版本不兼容会提示）
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521")
TEST_IMG = ROOT / "images" / "test"
TEST_LBL = ROOT / "labels" / "test"
IMGSZ = 1536  # 与训练一致（长边 1536，按原比例）

MODELS = [
    ("yolo26", Path(r"D:\yolov11\runs_pose\ice80_yolo26\weights\best.pt")),
    ("yolo11", Path(r"D:\yolov11\runs_pose\ice80_yolo11\weights\best.pt")),
]


def load_gt(lbl_path, w, h):
    """返回 [(box xyxy, [(x,y,v) x3])]（像素坐标）"""
    out = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 14:
            continue
        cx, cy, bw, bh = map(float, f[1:5])
        box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        kps = []
        for k in range(3):
            x, y, v = float(f[5 + k * 3]) * w, float(f[6 + k * 3]) * h, int(float(f[7 + k * 3]))
            kps.append((x, y, v))
        out.append((box, kps))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(tag, wpath):
    if not wpath.exists():
        print(f"[{tag}] 缺少权重 {wpath}，跳过")
        return
    from ultralytics import YOLO

    model = YOLO(str(wpath))
    conf_thr = 0.25
    errs, rels = [], []
    n_gt = n_pred = n_match = 0
    per_img = []

    from PIL import Image

    for img_path in sorted(TEST_IMG.glob("*.jpg")):
        lbl = TEST_LBL / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        gts = load_gt(lbl, W, H)
        n_gt += len(gts)

        r = model.predict(source=str(img_path), imgsz=IMGSZ, conf=conf_thr, verbose=False)[0]
        pbox = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        pkp = r.keypoints.xy.cpu().numpy() if r.keypoints is not None and len(r.keypoints) else np.zeros((0, 3, 2))
        n_pred += len(pbox)

        used = set()
        for box, kps in gts:
            cand = [(iou(box, pb), i) for i, pb in enumerate(pbox) if i not in used]
            if not cand:
                continue
            best = max(cand)
            if best[0] < 0.3:
                continue
            pi = best[1]
            used.add(pi)
            n_match += 1
            bh = max(1.0, box[3] - box[1])
            for k, (x, y, v) in enumerate(kps):
                if v == 0:
                    continue
                d = float(np.hypot(pkp[pi][k][0] - x, pkp[pi][k][1] - y))
                errs.append(d)
                rels.append(d / bh)
        per_img.append((img_path.stem, len(gts), len(pbox)))

    if not errs:
        print(f"[{tag}] 没有可评估的点")
        return
    rels = np.array(rels)
    print(f"\n===== {tag} =====")
    print(f"权重 {wpath}")
    print(f"真实框 {n_gt}  预测框 {n_pred}  匹配成功 {n_match}")
    print(f"关键点平均误差 {np.mean(errs):.2f} px  中位 {np.median(errs):.2f} px")
    print(f"按框高归一化  平均 {np.mean(rels) * 100:.2f}%  中位 {np.median(rels) * 100:.2f}%")
    print(f"PCK@5%  {np.mean(rels < 0.05) * 100:.1f}%")
    print(f"PCK@10% {np.mean(rels < 0.10) * 100:.1f}%")
    print("逐图（真实框/预测框）:", ", ".join(f"{s}:{g}/{p}" for s, g, p in per_img))


if __name__ == "__main__":
    for tag, w in MODELS:
        try:
            evaluate(tag, w)
        except Exception as e:  # 版本不兼容时不影响另一个
            print(f"[{tag}] 评估失败: {type(e).__name__}: {e}")
    sys.exit(0)
