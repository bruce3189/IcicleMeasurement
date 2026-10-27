"""对 400 张池子里尚未标注的 320 张跑预测，并把旧框（old_data 里只有 ice 矩形的那批）一起比对。

输出: D:\yolov11\runs_pose\pred_320\raw.json
"""
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

POOL = Path(r"C:\Users\Administrator\Desktop\images")
LABELED_ROOT = Path(r"C:\Users\Administrator\Desktop\Iceicle_Easy80_YOLOPose_521\images")
OLD_ROOT = Path(r"C:\Users\Administrator\Desktop\old_data\Iceicle_Labelme_YOLOPose_521")
WEIGHTS = r"D:\yolov11\runs_pose\ice80_yolo26\weights\best.pt"
OUT = Path(r"D:\yolov11\runs_pose\pred_320")
OUT.mkdir(parents=True, exist_ok=True)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def main():
    labeled = {p.stem for p in LABELED_ROOT.glob("*/*.jpg")}
    imgs = sorted(POOL.glob("*.jpg"))
    todo = [p for p in imgs if p.stem not in labeled]
    print("池子 %d，已标 %d，待整 %d" % (len(imgs), len(labeled), len(todo)))

    model = YOLO(WEIGHTS)
    recs = []
    for i, p in enumerate(todo, 1):
        r = model.predict(source=str(p), imgsz=1536, conf=0.25, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        confs = r.boxes.conf.cpu().numpy().tolist()
        kpts = r.keypoints.xy.cpu().numpy().tolist()

        old_boxes = []
        oj = next(OLD_ROOT.rglob(p.stem + ".json"), None)
        if oj is not None:
            d = json.loads(oj.read_text(encoding="utf-8-sig"))
            for s in d["shapes"]:
                if s["shape_type"] == "rectangle":
                    (ax, ay), (bx, by) = s["points"]
                    old_boxes.append([min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)])

        recs.append({"img": p.stem, "w": r.orig_shape[1], "h": r.orig_shape[0],
                     "boxes": boxes, "confs": confs, "kpts": kpts, "old": old_boxes})
        if i % 40 == 0:
            print("  %d/%d" % (i, len(todo)), flush=True)

    (OUT / "raw.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")

    n_new = sum(len(x["boxes"]) for x in recs)
    n_old = sum(len(x["old"]) for x in recs)
    # 旧框里新框没覆盖的部分
    old_only = 0
    new_only = 0
    for x in recs:
        nb, ob = x["boxes"], x["old"]
        for o in ob:
            if not nb or max(iou(o, b) for b in nb) < 0.3:
                old_only += 1
        for b in nb:
            if not ob or max(iou(o, b) for o in ob) < 0.3:
                new_only += 1
    print("新模型框 %d，旧框 %d" % (n_new, n_old))
    print("旧框里新模型没匹配上（可能是新模型漏检）: %d 个" % old_only)
    print("新框里旧框没有（新模型多检出）: %d 个" % new_only)
    print("每图框数: 新 中位 %.0f 最多 %d | 旧 中位 %.0f 最多 %d" % (
        np.median([len(x["boxes"]) for x in recs]), max(len(x["boxes"]) for x in recs),
        np.median([len(x["old"]) for x in recs]), max(len(x["old"]) for x in recs)))


if __name__ == "__main__":
    main()
