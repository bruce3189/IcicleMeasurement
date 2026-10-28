# tip 加权训练 · 232 张合格数据集（2026-10-23）

沿用 `..\tip_refine` 的实验框架（YOLO26 pose + 逐点关键点加权），只把数据换成新的 232 张合格集。

## 数据

- 快照：`data_snapshot/`（train 169 / val 38 / test 25），由 `snapshot_232.py` 从
  `C:\Users\Administrator\Desktop\Iceicle_数据集_400_20261023` 复制而来，附 SHA256 清单 `source_manifest.json`
- 与旧 Easy80 快照的区别：① 图片 80 → 232；② 划分改为同批连续帧 4 张一块整块分配（原来含相邻帧泄漏）
- 标签：1633 框，三点齐 1139，无关键点的框已剔除，未归属点 17

## 起点权重

`D:/yolov11/runs_pose/ice80_yolo26/weights/best.pt`（旧 80 张训练出的 pose 模型），低学习率微调。

## 本次实验

```
train.py --tip-weight 3 --epochs 10 --name tip3_232_b1_10ep --run
```

- tip : base_left : base_right = 3 : 1 : 1，均值归一化为 1.8 : 0.6 : 0.6（总 pose 损失量级不变）
- OKS 定位与 RLE 两项同时逐点加权；框损失、可见性损失、sigma 不变；one2many/one2one 两分支都加权
- 1280、rect、CPU、batch 1、workers 0、AdamW、lr 2e-4、seed 42、mosaic/翻转关闭、val 批量固定 1
- 结束后自动用 `evaluate.py` 在 val 上评估 `last.pt`，写到 `experiments/tip3_232_b1_10ep/raw_val`

## 怎么看结果

1. `evaluate.py` 的 `tip_success10_all_gt`（有 tip 真值里既匹配到框、又落在框高 10% 内的比例，漏匹配算失败）
2. tip 二维误差中位数与 P90，尤其长目标（>1200px）
3. 框 IoU、框底误差、base_left/right 是否退步；漏匹配/多余框是否变多
4. 要和"不加权对照组"比才成立——同一数据、同一轮数、同一学习率下跑 `--tip-weight 1`

日志：`tip3_232_b1_10ep.log`
