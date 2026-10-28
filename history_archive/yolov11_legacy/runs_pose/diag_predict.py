# -*- coding: utf-8 -*-
"""用 best.pt 对 test 集做预测，并生成「真值 / 预测」并排对照图 + 逐框统计"""
import sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521")
OUT = Path(r"C:\Users\Administrator\Desktop\冰凌_预测对比_20261022")
OUT.mkdir(exist_ok=True)
W_PT = r"D:\yolov11\runs_pose\ice80_yolo26\weights\best.pt"
FONT = r"C:\Windows\Fonts\msyh.ttc"
IMGSZ = 1536
CONF = 0.25

GT_COL = (0, 200, 120)      # 真值框 绿
GT_KPT = (0, 90, 255)       # 真值关键点 蓝
PD_COL = (255, 40, 40)      # 预测框 红
PD_KPT = (255, 190, 0)      # 预测关键点 橙黄
NAMES = ["tip", "base_l", "base_r"]


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


def main():
    model = YOLO(W_PT)
    f_small = ImageFont.truetype(FONT, 26)
    rows = []
    for img_path in sorted((ROOT / "images" / "test").glob("*.jpg")):
        name = img_path.stem
        lbl = ROOT / "labels" / "test" / (name + ".txt")
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gts, gk = load_gt(lbl, W, H)
        r = model.predict(source=str(img_path), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy()
        pk = r.keypoints.xy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()

        # 贪心匹配：按 IoU 从高到低
        cand = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(pb)),
                      reverse=True)
        used_g, used_p, pairs = set(), set(), []
        for v, gi, pi in cand:
            if v < 0.3 or gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi); pairs.append((gi, pi, v))

        for gi, pi, v in pairs:
            gh = gts[gi][3] - gts[gi][1]
            e = [float(np.hypot(pk[pi][k][0] - gk[gi][k][0], pk[pi][k][1] - gk[gi][k][1])) for k in range(3)]
            rows.append(dict(img=name, gi=gi, bh=gh, iou=v,
                             tip=e[0], bl=e[1], br=e[2], tip_pct=100 * e[0] / gh,
                             conf=float(conf[pi])))

        # ---- 画对照图：左真值 右预测 ----
        crop_top = int(max(0, min([g[1] for g in gts] + [p[1] for p in pb]) - 60))
        crop_bot = int(min(H, max([g[3] for g in gts] + [p[3] for p in pb]) + 60))
        left, right = img.copy(), img.copy()
        dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
        for g, k in zip(gts, gk):
            dl.rectangle(g, outline=GT_COL, width=8)
            for (x, y) in k:
                dl.ellipse([x - 12, y - 12, x + 12, y + 12], fill=GT_KPT, outline=(255, 255, 255), width=3)
        for i in range(len(pb)):
            p = pb[i]
            dr.rectangle(p, outline=PD_COL, width=8)
            for k in range(3):
                x, y = pk[i][k]
                dr.ellipse([x - 12, y - 12, x + 12, y + 12], fill=PD_KPT, outline=(0, 0, 0), width=3)
        for im, txt in ((left, "真值 GT  %d 框" % len(gts)), (right, "预测 PRED  %d 框" % len(pb))):
            d = ImageDraw.Draw(im)
            d.rectangle([0, crop_top, 520, crop_top + 46], fill=(20, 20, 20))
            d.text((12, crop_top + 8), txt, font=f_small, fill=(255, 255, 255))
        both = Image.new("RGB", (left.width * 2 + 12, crop_bot - crop_top), (255, 255, 255))
        both.paste(left.crop((0, crop_top, W, crop_bot)), (0, 0))
        both.paste(right.crop((0, crop_top, W, crop_bot)), (left.width + 12, 0))
        # 缩放到高 1500
        sc = 1500 / both.height
        if sc < 1:
            both = both.resize((int(both.width * sc), 1500), Image.LANCZOS)
        both.save(OUT / (name + "_对比.jpg"), quality=88)

    # ---- 统计 ----
    print("%-10s %6s %6s %5s | %8s %8s %7s" % ("图", "框高", "IoU", "conf", "tip误差", "tip/框高", "base误差"))
    for r in sorted(rows, key=lambda x: -x["tip_pct"]):
        print("%-10s %6.0f %6.2f %5.2f | %7.0fpx %7.1f%% %6.0fpx" % (
            r["img"], r["bh"], r["iou"], r["conf"], r["tip"], r["tip_pct"],
            (r["bl"] + r["br"]) / 2))
    a = np.array([[r["tip"], r["bl"], r["br"], r["bh"], r["tip_pct"]] for r in rows])
    print("\n匹配框 %d 个：tip 中位误差 %.0fpx（%.1f%% 框高） | base 中位 %.0fpx | IoU 中位 %.2f" % (
        len(rows), np.median(a[:, 0]), np.median(a[:, 4]), np.median(a[:, 1:3]), np.median([r["iou"] for r in rows])))
    print("tip 误差 >10%%框高 的框数: %d / %d" % (int((a[:, 4] > 10).sum()), len(rows)))
    print("按框高分组 tip%%：")
    for lo, hi in ((0, 700), (700, 1200), (1200, 2000)):
        sel = [r["tip_pct"] for r in rows if lo <= r["bh"] < hi]
        if sel:
            print("  框高 %4d~%4dpx: n=%2d  tip误差中位 %.1f%%" % (lo, hi, len(sel), np.median(sel)))


if __name__ == "__main__":
    main()
