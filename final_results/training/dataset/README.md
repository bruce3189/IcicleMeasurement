# 冰凌两点检测数据集 · 78 张（2026-10-23）

由"根根分明"干净子集（78 张，源 `Desktop\冰凌_根根分明子集_20261023`）转换而来，
关键点从三点（tip / base_left / base_right）合并为**两点**：**tip（下端尖端）+ base（根部中点）**。

## 转换规则

- `base = (base_left + base_right) / 2`（根部中点）
- base 可见性：base_left 和 base_right **都**可见才为 2，否则为 0
- tip 直接沿用原标签；框坐标不变
- tip ↔ base 连线 = 冰凌中轴线，长度 ≈ 冰凌长度

## 标签格式（每行）

```
class cx cy w h  x1 y1 v1   x2 y2 v2
                 ↑tip        ↑base
```

归一化 0~1，v=2 可见 / v=0 缺失。`kpt_shape: [2, 3]`、`flip_idx: [0, 1]`（水平翻转不交换两点）。

## 规模

| split | 图 | 框 |
|---|---|---|
| train | 57 | — |
| val | 16 | — |
| test | 5 | — |
| **合计** | **78** | **590** |

- tip 可见 588（99.7%）
- base 可见 557（94.4%，即两 base 点齐全的）
- **两点齐 556（94.2%）**，全缺 1

## 训练

```bash
yolo pose train data=C:\Users\Administrator\Desktop\冰凌_两点检测_78_20261023\dataset.yaml model=yolo11n-pose.pt imgsz=1280
```

## 注意

- 本目录是干净子集的两点版，**不含**复杂重叠/框宽于冰凌的图
- 源三点版改动后此处不会自动更新，需重跑 `D:\ipa_work\build_2kpt.py`
- 若此前笔记/随笔按三点口径写了"三点齐才算体积"等规则，两点口径下应为"tip + base 两点齐"
