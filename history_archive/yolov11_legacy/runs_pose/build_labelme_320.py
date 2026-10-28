"""把 320 张待标注图整理成 labelme 可直接打开的工作目录。

框来源：yolo26 best.pt 预测（IoU>0.65 去重），并补入旧标注里有、新模型没有的框。
点来源：base_left / base_right 用模型预测点；tip 用（模型 tip.x, 框底 - 3.7% 框高）。
图片用硬链接，不占额外空间；跨卷失败时复制。
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np

RAW = Path(r"D:\yolov11\runs_pose\pred_320\raw.json")
POOL = Path(r"C:\Users\Administrator\Desktop\images")
OUT = Path(r"C:\Users\Administrator\Desktop\Iceicle_待标320_20261022")
IMGDIR = OUT / "images"

TIP_Y_RATIO = 0.963      # tip 落在框高的 96.3%
BASE_L = (0.235, 0.029)  # 旧框补漏时按统计中位生成点
BASE_R = (0.754, 0.028)
DEDUP_IOU = 0.65


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def dedup(boxes, confs):
    order = sorted(range(len(boxes)), key=lambda i: -confs[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < DEDUP_IOU for j in keep):
            keep.append(i)
    keep.sort()
    return keep


def rect_shape(gid, box, desc=""):
    x1, y1, x2, y2 = box
    return {"label": "ice", "points": [[x1, y1], [x2, y2]], "group_id": gid,
            "description": desc, "shape_type": "rectangle", "flags": {"occluded": False}}


def point_shape(gid, label, x, y, desc=""):
    return {"label": label, "points": [[x, y]], "group_id": gid,
            "description": desc, "shape_type": "point", "flags": {"occluded": False}}


def main():
    recs = json.loads(RAW.read_text(encoding="utf-8"))
    for d in (IMGDIR, OUT / "预览"):
        d.mkdir(parents=True, exist_ok=True)

    report = []
    n_box = n_pt = n_old_added = n_dedup = n_lowconf = 0
    link_ok = copy_ok = 0

    for rec in recs:
        boxes = [list(map(float, b)) for b in rec["boxes"]]
        confs = [float(c) for c in rec["confs"]]
        kpts = rec["kpts"]
        W, H = rec["w"], rec["h"]

        keep = dedup(boxes, confs)
        n_dedup += len(boxes) - len(keep)
        items = [(boxes[i], confs[i], kpts[i], False) for i in keep]

        # 旧框补漏：新模型完全没覆盖到的旧框
        kept_boxes = [boxes[i] for i in keep]
        for ob in rec["old"]:
            ob = list(map(float, ob))
            if not kept_boxes or max(iou(ob, b) for b in kept_boxes) < 0.3:
                if max([iou(ob, b) for b in kept_boxes] or [0]) < 0.3:
                    items.append((ob, 0.0, None, True))
                    kept_boxes.append(ob)
                    n_old_added += 1

        # 按置信度降序编 gid
        items.sort(key=lambda t: -t[1])
        shapes = []
        pts = []
        for gid, (b, c, kp, from_old) in enumerate(items, 1):
            x1, y1, x2, y2 = b
            w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            def clamp_x(v):
                return min(max(v, x1 + 1.0), x2 - 1.0)

            if kp is None:
                blx = x1 + BASE_L[0] * w
                brx = x1 + BASE_R[0] * w
                tipx = (x1 + x2) / 2
            else:
                blx, brx, tipx = kp[1][0], kp[2][0], kp[0][0]
            # 纵向统一按你的标注约定（base 贴框顶、tip 贴框底），横向保留模型的判断
            bl = (clamp_x(blx), y1 + BASE_L[1] * h)
            br = (clamp_x(brx), y1 + BASE_R[1] * h)
            tipx = clamp_x(tipx)
            tipy = y1 + TIP_Y_RATIO * h

            desc = "旧框补充,请复核" if from_old else ("conf=%.2f" % c)
            if c and c < 0.40:
                n_lowconf += 1
            shapes.append(rect_shape(gid, b, desc))
            pts.append(point_shape(gid, "tip", tipx, tipy, desc))
            pts.append(point_shape(gid, "base_left", bl[0], bl[1], desc))
            pts.append(point_shape(gid, "base_right", br[0], br[1], desc))

        doc = {"version": "6.3.1", "flags": {}, "shapes": shapes + pts,
               "imagePath": rec["img"] + ".jpg", "imageData": None,
               "imageHeight": H, "imageWidth": W}
        (IMGDIR / (rec["img"] + ".json")).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        src = POOL / (rec["img"] + ".jpg")
        dst = IMGDIR / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
                link_ok += 1
            except OSError:
                shutil.copy2(src, dst)
                copy_ok += 1

        n_box += len(shapes)
        n_pt += len(pts)
        report.append((rec["img"], len(shapes), len(pts), sum(1 for t in items if t[1] and t[1] < 0.4),
                       sum(1 for t in items if t[3])))

    print("生成 %d 张：框 %d 个，点 %d 个" % (len(recs), n_box, n_pt))
    print("去重合并 %d 个框 | 旧框补充 %d 个 | 低置信(<0.40) %d 个" % (n_dedup, n_old_added, n_lowconf))
    print("图片：硬链接 %d，复制 %d" % (link_ok, copy_ok))

    lines = ["每图检出情况（按需复核，重点是低置信与旧框补充）", "=" * 72,
             "%-12s %5s %5s %8s %8s" % ("图片", "框数", "点数", "低置信", "旧框补")]
    for r in report:
        lines.append("%-12s %5d %5d %8d %8d" % r)
    (OUT / "逐图清单.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("清单 ->", OUT / "逐图清单.txt")


if __name__ == "__main__":
    main()
