# -*- coding: utf-8 -*-
"""对比 tip 加权(×3) 与不加权两版权重：test 集上 tip/base 分开、按框高分档。"""
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

ROOT = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521")
MODELS = {
    "不加权": r"D:\yolov11\runs_pose\ice80_yolo26\weights\best.pt",
    "tip×3": r"D:\yolov11\runs_pose\ice80_yolo26_tipw\weights\best.pt",
}
CONF = 0.25


def load_gt(lbl, W, H):
    B, K = [], []
    for line in open(lbl, encoding="utf-8"):
        f = line.split()
        if len(f) < 14:
            continue
        cx, cy, bw, bh = map(float, f[1:5])
        B.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        K.append([(float(f[5 + 3 * k]) * W, float(f[6 + 3 * k]) * H, float(f[7 + 3 * k])) for k in range(3)])
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


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def run(model_path, tag):
    m = YOLO(model_path)
    res = {"tip_px": [], "tip_ratio": [], "base_px": [], "base_ratio": [],
           "rule_px": [], "rule_ratio": [], "iou": [], "nbox": 0, "npred": 0, "bins": {}}
    for img in sorted((ROOT / "images" / "test").glob("*.jpg")):
        lbl = ROOT / "labels" / "test" / (img.stem + ".txt")
        W, H = Image.open(img).size
        B, K = load_gt(lbl, W, H)
        r = m.predict(source=str(img), imgsz=1536, conf=CONF, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy()
        pk = r.keypoints.xy.cpu().numpy()
        res["nbox"] += len(B); res["npred"] += len(pb)
        for gi, pi, v in match(B, pb):
            g, p = B[gi], pb[pi]
            bh = p[3] - p[1]
            res["iou"].append(v)
            tip = K[gi][0]
            if tip[2] != 0:
                et = abs(pk[pi][0][1] - tip[1])
                res["tip_px"].append(et); res["tip_ratio"].append(et / bh)
                # 规则推算：x=两 base 中点，y=框底-4%
                ry = p[3] - 0.04 * bh
                rx = (pk[pi][1][0] + pk[pi][2][0]) / 2
                er = np.hypot(rx - tip[0], ry - tip[1])
                res["rule_px"].append(er); res["rule_ratio"].append(er / bh)
                # 分档
                k = ">1200" if bh > 1200 else ("700-1200" if bh >= 700 else "<700")
                b = res["bins"].setdefault(k, {"tip": [], "rule": []})
                b["tip"].append(et / bh); b["rule"].append(er / bh)
            for kk in (1, 2):
                if K[gi][kk][2] == 0:
                    continue
                e = np.hypot(pk[pi][kk][0] - K[gi][kk][0], pk[pi][kk][1] - K[gi][kk][1])
                res["base_px"].append(e); res["base_ratio"].append(e / bh)
    print(f"\n===== {tag} =====")
    print(f"GT框 {res['nbox']} / 预测框 {res['npred']} / 匹配 {len(res['iou'])} | 框IoU中位 {np.median(res['iou']):.3f}")
    for k in ("tip", "base", "rule"):
        a, b = np.array(res[k + "_px"]), np.array(res[k + "_ratio"])
        print(f"{k:5s} 误差 中位 {pct(a,50):5.1f}px ({pct(b,50)*100:4.1f}%框高)  均值 {a.mean():5.1f}px  p90 {pct(a,90):6.1f}px ({pct(b,90)*100:4.1f}%)")
    for k in (">1200", "700-1200", "<700"):
        if k in res["bins"]:
            b = res["bins"][k]
            print(f"  {k:9s} n={len(b['tip']):3d}  模型tip {np.median(b['tip'])*100:5.1f}%  规则tip {np.median(b['rule'])*100:5.1f}%")
    return res


if __name__ == "__main__":
    out = {t: run(p, t) for t, p in MODELS.items()}
    print("\n===== 差值（tip×3 − 不加权）=====")
    a, b = out["不加权"], out["tip×3"]
    print(f"tip 中位 {pct(np.array(a['tip_px']),50):.1f} -> {pct(np.array(b['tip_px']),50):.1f} px")
    print(f"tip 均值 {np.mean(a['tip_px']):.1f} -> {np.mean(b['tip_px']):.1f} px")
    print(f"tip p90  {pct(np.array(a['tip_px']),90):.1f} -> {pct(np.array(b['tip_px']),90):.1f} px")
    print(f"base 中位 {pct(np.array(a['base_px']),50):.1f} -> {pct(np.array(b['base_px']),50):.1f} px")
