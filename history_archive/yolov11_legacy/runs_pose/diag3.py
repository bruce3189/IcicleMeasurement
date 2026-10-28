# -*- coding: utf-8 -*-
"""诊断 + 修正：
1) 比较 imgsz 1280/1536 的框与 tip 精度
2) 生成三联对照图：真值 | 模型原始输出 | 修正后（去重 + tip 贴框底）
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521")
OUT = Path(r"C:\Users\Administrator\Desktop\冰凌_预测对比_20261022")
OUT.mkdir(exist_ok=True)
MODEL = YOLO(r"D:\yolov11\runs_pose\ice80_yolo26\weights\best.pt")
FONT = r"C:\Windows\Fonts\msyh.ttc"
CONF = 0.25


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


def match(B, pb):
    cand = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(B) for pi, p in enumerate(pb)), reverse=True)
    ug, up, pairs = set(), set(), []
    for v, gi, pi in cand:
        if v < 0.3 or gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi); pairs.append((gi, pi, v))
    return pairs


def predict(path, sz):
    r = MODEL.predict(source=str(path), imgsz=sz, conf=CONF, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def correct(boxes, kps, confs, dedup_iou=0.55):
    """去重 + tip 贴框底（x 取两个 base 的中点）"""
    order = np.argsort(-confs)
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < dedup_iou for j in keep):
            keep.append(i)
    out_b, out_k = [], []
    for i in keep:
        b = boxes[i].copy()
        k = kps[i].copy()
        h = b[3] - b[1]
        k[0] = [(k[1][0] + k[2][0]) / 2, b[3] - 0.04 * h]
        out_b.append(b); out_k.append(k)
    return np.array(out_b), np.array(out_k)


def metrics(sz):
    box_iou, tip_raw, tip_fix, n_pred, n_gt = [], [], [], 0, 0
    for ip in sorted((ROOT / "images" / "test").glob("*.jpg")):
        im = Image.open(ip); W, H = im.size
        B, K = load_gt(ROOT / "labels" / "test" / (ip.stem + ".txt"), W, H)
        pb, pk, cf = predict(ip, sz)
        n_gt += len(B); n_pred += len(pb)
        for gi, pi, v in match(B, pb):
            box_iou.append(v)
            tip_raw.append(abs(pk[pi][0][1] - K[gi][0][1]))
            tip_fix.append(abs((pb[pi][3] - 0.04 * (pb[pi][3] - pb[pi][1])) - K[gi][0][1]))
    return (np.median(box_iou), np.median(tip_raw), np.median(tip_fix), n_pred, n_gt)


def draw(img, boxes, kps, bcol, kcol, title, font):
    im = img.copy(); d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle(b, outline=bcol, width=8)
    for k in kps:
        for (x, y) in k:
            d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=kcol, outline=(255, 255, 255), width=3)
    d.rectangle([0, 0, 620, 46], fill=(20, 20, 20))
    d.text((10, 8), title, font=font, fill=(255, 255, 255))
    return im


def main():
    print("imgsz 对比：")
    for sz in (1280, 1536):
        bi, tr, tf, npd, ngt = metrics(sz)
        print("  %4d  框IoU中位 %.2f | tip误差 原始 %.0fpx → 修正 %.0fpx | 预测框 %d / 真值 %d" % (
            sz, bi, tr, tf, npd, ngt))

    print("\n生成三联对照图（1536 + 修正）")
    font = ImageFont.truetype(FONT, 26)
    for ip in sorted((ROOT / "images" / "test").glob("*.jpg")):
        img = Image.open(ip).convert("RGB"); W, H = img.size
        B, K = load_gt(ROOT / "labels" / "test" / (ip.stem + ".txt"), W, H)
        pb, pk, cf = predict(ip, 1536)
        cb, ck = correct(pb, pk, cf)
        cols = [
            draw(img, B, K, (0, 200, 120), (0, 90, 255), "真值 GT %d 框" % len(B), font),
            draw(img, pb, pk, (255, 40, 40), (255, 190, 0), "模型原始 %d 框" % len(pb), font),
            draw(img, cb, ck, (230, 0, 160), (0, 200, 255), "修正后 %d 框（去重+tip贴框底）" % len(cb), font),
        ]
        y0 = int(max(0, min([g[1] for g in B] + [b[1] for b in pb]) - 70))
        y1 = int(min(H, max([g[3] for g in B] + [b[3] for b in pb]) + 70))
        canvas = Image.new("RGB", (W * 3 + 24, y1 - y0), (255, 255, 255))
        for i, c in enumerate(cols):
            canvas.paste(c.crop((0, y0, W, y1)), (i * (W + 12), 0))
        sc = 1800 / canvas.height
        if sc < 1:
            canvas = canvas.resize((int(canvas.width * sc), 1800), Image.LANCZOS)
        canvas.save(OUT / (ip.stem + "_三联.jpg"), quality=86)
    print("已写入", OUT)


if __name__ == "__main__":
    main()
