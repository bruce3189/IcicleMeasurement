# -*- coding: utf-8 -*-
"""
YOLO 推理 + 重叠框筛选（细线版，三种模式）

功能说明
--------
通过顶部 FILTER_MODE 切换三种模式：

  "off"      原始模式：行为与最早的 mypredict.py 完全一致，
             Ultralytics 原生 save=True / save_txt=True 输出（runs/detect/predict*）。

  "largest"  互相重叠的检测框只保留面积最大的那一个（IoU+IoM 双判据），
             适合"大框套小框 / 同一目标多个重复框"的场景。

  "soft_nms" Soft-NMS：重叠框不删除，按重叠程度衰减置信度，衰减后仍过线的保留，
             适合一大堆目标挤在一起的密集场景，尽量不漏被压住的真实目标。

两种加强模式都会按"细线"风格重绘标注图（线宽 1 像素），并导出 YOLO 格式 txt。

为什么不能直接用 save=True（加强模式）
--------------------------
YOLO 的 model.predict(save=True) 是在推理阶段就把图画好写盘的，
那时后处理还没做，所以图里会保留重复框。因此这里关掉 save，
改为：先筛选框 → 再手动 r.plot() → 自己 cv2.imwrite。

为什么自带 NMS 不够
-------------------
YOLO 的 NMS 保留的是"置信度最高"的框，不是"面积最大"的框，
且硬阈值会把与真目标重叠的框直接删掉。
所以把 predict 的 iou 参数放宽（0.7），让重叠框尽量都留下来，
再由 keep_largest_boxes()（按面积裁决）或 soft_nms_boxes()（按分数衰减裁决）处理。
"""

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent


import cv2
import torch
from ultralytics import YOLO

# ============================ 开关与可调参数 ============================

# ---- 总开关：重叠框筛选模式 ----
# "off"      = 关闭筛选，完全回到原始 myredict 行为（Ultralytics 原生输出）
# "largest"  = 互相重叠的框只保留面积最大的那个（IoU + IoM 双判据，原加强功能）
# "soft_nms" = Soft-NMS：重叠框不删除，按重叠程度衰减置信度，衰减后仍达标的保留
#              （适合一大堆目标挤在一起的场景，保留更多被压住的真实目标）
FILTER_MODE = "soft_nms"

MODEL_PATH = str(PROJECT_ROOT / 'runs/detect/train10/weights/best.pt')  # 训练好的权重
SOURCE = str(PROJECT_ROOT.parent / 'Measurement/assets/rgb')                 # 待推理的图片文件夹
OUT_DIR = str(PROJECT_ROOT / 'runs/detect/predict_largest')               # 输出目录（图 + labels）
# （"off" 模式不走 OUT_DIR，结果自动存到 runs/detect/predict*，和原来一样）

OVERLAP_THR = 0.5     # "largest" 模式：重叠判定阈值，0.3 压得狠，0.7 只压几乎重合的
SOFT_NMS_SIGMA = 0.5  # "soft_nms" 模式：高斯衰减系数，越小衰减越狠（压得越狠）
LINE_WIDTH = 1        # 画框线宽（像素），1 = 细线；0 会退化成默认的 2~3 像素
FONT_SIZE = 9         # 标签文字大小（像素级），越小越不抢戏
CONF_THR = 0.25       # 置信度阈值：低于此值的框直接丢弃（soft_nms 里是衰减后的最终线）
NMS_IOU = 0.7         # 预测阶段 NMS 的 IoU 阈值，放宽以便把重叠框交给后处理裁决

# ==================================================================


def keep_largest_boxes(boxes, thr=OVERLAP_THR):
    """
    在互相重叠的框中只保留面积最大的那个。

    参数
    ----
    boxes : ultralytics.engine.results.Boxes
        单张图的检测框集合，坐标已是原图像素坐标（xyxy）。
    thr : float
        重叠判定阈值，同时作用于两种判据：
          - IoU  = 交集 / 并集          （常规重叠）
          - IoM  = 交集 / 小框面积       （大框完全套住小框的嵌套情况）
        任一判据超过 thr，就认为两个框"重叠"，保留大的、丢弃小的。

    返回
    ----
    torch.BoolTensor
        长度等于 len(boxes) 的布尔掩码，True 表示该框保留。

    算法
    ----
    贪心：按面积从大到小遍历，遇到"还没被压制"的框就保留，
    并把它压制范围（所有与它重叠的框）全部标记为丢弃。
    因为是从大到小，所以每一组重叠框里第一个被处理的一定是最大的那个，
    它会把同组的其他框全部压掉 —— 天然实现"只留最大"。
    """
    n = len(boxes)
    # 0 个或 1 个框，没有可比较的对象，全保留
    if n <= 1:
        return torch.ones(n, dtype=torch.bool)

    xyxy = boxes.xyxy                                  # (N, 4)：左上 x,y + 右下 x,y
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # 每个框的面积 (N,)

    # ---- 两两计算交集面积，用广播一次性算出 N×N 矩阵 ----
    # 交集矩形的左上角 = 两框左上角取大；右下角 = 两框右下角取小
    lt = torch.max(xyxy[:, None, :2], xyxy[None, :, :2])   # (N, N, 2)
    rb = torch.min(xyxy[:, None, 2:], xyxy[None, :, 2:])   # (N, N, 2)
    wh = (rb - lt).clamp(min=0)                            # 负宽度说明不相交，截断为 0
    inter = wh[..., 0] * wh[..., 1]                        # (N, N) 交集面积

    # 判据一：IoU = 交集 / 并集（并集 = A + B - 交集）
    iou = inter / (areas[:, None] + areas[None, :] - inter).clamp(min=1)
    # 判据二：IoM = 交集 / 较小框面积。嵌套时 IoU 会被大框"稀释"得很小，
    #         IoM 则始终接近 1，能把"大框套小框"正确识别为重叠
    iom = inter / torch.min(areas[:, None], areas[None, :]).clamp(min=1)

    overlap = (iou > thr) | (iom > thr)   # (N, N) 布尔矩阵：i 与 j 是否重叠

    keep = torch.ones(n, dtype=torch.bool)     # 初始全部保留
    for i in areas.argsort(descending=True).tolist():   # 面积从大到小
        if not keep[i]:
            continue            # 已被更大的框压掉了，跳过
        keep &= ~overlap[i]     # 压制所有与 i 重叠的框（含互相之间的传递压制）
        keep[i] = True          # 上面那行会把自己也置 False，这里补回来
    return keep


def soft_nms_boxes(boxes, sigma=SOFT_NMS_SIGMA, score_thr=CONF_THR):
    """
    Soft-NMS：重叠框不删除，只按重叠程度衰减置信度（高斯衰减）。

    参数
    ----
    boxes : ultralytics.engine.results.Boxes
        单张图的检测框集合。
    sigma : float
        高斯衰减系数，s_i ← s_i * exp(-IoU² / sigma)。越小衰减越狠。
    score_thr : float
        最终分数线：衰减后仍高于它的框保留，其余丢弃。

    返回
    ----
    torch.LongTensor
        被保留框的原始索引（按置信度从高到低的挑选顺序）。

    算法
    ----
    每轮挑出当前分数最高的框保留，然后给所有剩余框按与它的 IoU 打折
    （完全重叠→分数趋近 0，只是搭到边→几乎不打折），再进入下一轮。
    与硬 NMS 的本质区别：被大框罩住的小框分数被压低但不归零，
    只要衰减后仍过线，就作为独立目标活下来 —— 挤在一起的真目标不会被误杀。
    """
    n = len(boxes)
    if n <= 1:
        return torch.arange(n)

    idx = torch.arange(n)                     # 原始索引，随淘汰同步收缩
    xyxy = boxes.xyxy.clone().float()
    scores = boxes.conf.clone().float()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    keep = []
    while scores.numel():
        m = int(scores.argmax())              # 当前分数最高的框
        keep.append(int(idx[m]))

        # 它与所有框（含自身）的 IoU
        lt = torch.max(xyxy[m, :2], xyxy[:, :2])
        rb = torch.min(xyxy[m, 2:], xyxy[:, 2:])
        inter = ((rb - lt).clamp(min=0)).prod(dim=1)
        iou = inter / (areas[m] + areas - inter).clamp(min=1)

        # 高斯衰减：与选中框越重叠，分数掉得越多，但不直接归零
        scores = scores * torch.exp(-(iou ** 2) / sigma)

        # 移除已选中的框，并淘汰衰减后过不了线的框
        alive = scores > score_thr
        alive[m] = False
        scores, xyxy, areas, idx = scores[alive], xyxy[alive], areas[alive], idx[alive]

    return torch.tensor(sorted(keep), dtype=torch.long)


def run_original(model):
    """原始模式（开关 False）：与最早的 mypredict.py 完全一致，Ultralytics 原生保存。"""
    results = model.predict(
        source=SOURCE,
        project=str(PROJECT_ROOT / 'runs/detect'),
        conf=CONF_THR,
        save=True,       # 原生画图并写盘
        show=False,
        save_txt=True,   # 原生导出 YOLO txt
        line_width=LINE_WIDTH,
        stream=True,     # 流式处理：逐张释放内存，避免整批结果堆在内存里导致 OOM
    )
    for r in results:
        pass  # 只为消费生成器，边跑边释放；如需统计可在此处理 r.boxes


def main():
    # 加载模型（只需一次，400 张图共用）
    model = YOLO(MODEL_PATH)

    if FILTER_MODE == "off":
        run_original(model)             # 原始：原生输出
    else:
        _run_enhanced_main(model)       # 加强：largest / soft_nms 两种筛选


def _run_enhanced_main(model):
    """加强模式主流程（FILTER_MODE = "largest" 或 "soft_nms"）。"""
    # stream=True 是必须的：否则所有图的结果会累积在内存里，
    # 每张图解码后约 18.9 MB，250 张左右就会 OOM（就是之前报的 18938880 字节）
    results = model.predict(
        source=SOURCE,
        project=str(PROJECT_ROOT / 'runs/detect'),
        conf=CONF_THR,
        iou=NMS_IOU,
        save=False,      # 关掉自动保存，见文件头说明
        show=False,
        save_txt=False,
        stream=True,     # 流式：处理完一张释放一张
    )

    # 准备输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    labels_dir = os.path.join(OUT_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    n_before = n_after = 0   # 统计：过滤前后的框总数

    for r in results:
        # 从完整路径里取出文件名（不含扩展名），作为输出文件名
        stem = os.path.splitext(os.path.basename(r.path))[0]

        # ---- 核心：按 FILTER_MODE 计算保留索引并应用 ----
        if FILTER_MODE == "soft_nms":
            keep_idx = soft_nms_boxes(r.boxes)          # Soft-NMS：分数衰减，挤在一起的目标尽量都留
        else:
            keep_idx = keep_largest_boxes(r.boxes).nonzero(as_tuple=True)[0]  # 只留最大框
        n_before += len(r.boxes)
        n_after += len(keep_idx)
        r.boxes = r.boxes[keep_idx]  # 直接替换结果里的框，后续 plot 就是过滤后的

        # ---- 画图：细线 + 小字号 ----
        # plot() 默认线宽会随图片尺寸自适应（通常 2~3 像素），显式传 1 才能画细
        annotated = r.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE, labels=True)
        cv2.imwrite(os.path.join(OUT_DIR, stem + ".jpg"), annotated)

        # ---- 导出 YOLO 格式 txt：cls cx cy w h（全部归一化到 0~1）----
        h, w = r.orig_shape
        lines = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h   # 中心点
            bw, bh = (x2 - x1) / w, (y2 - y1) / h           # 宽高
            lines.append(f"{int(b.cls[0])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"完成：过滤前 {n_before} 个框 → 过滤后 {n_after} 个框（去掉 {n_before - n_after} 个）")
    print(f"结果目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
