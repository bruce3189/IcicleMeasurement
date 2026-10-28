> 历史实验归档，不是当前采用模型。当前版本见 [最终成果](<../../../final_results/README.md>)。

# 历史预测、像素评估与人工复核

## 批量预测目录

统计为现存文件和标签非空行数；标签行数表示预测实例数量，不能当成正确检出数量。同名编号不保证同一数据或权重；没有参数证据的目录不补猜来源。

| 目录 | 图片 | 标签文件 | 预测实例 | 大小 MiB |
|---|---:|---:|---:|---:|
| [runs/detect/predict](<../runs/detect/predict>) | 400 | 400 | 3414 | 340.33 |
| [runs/detect/predict10](<../runs/detect/predict10>) | 400 | 400 | 3414 | 340.33 |
| [runs/detect/predict2](<../runs/detect/predict2>) | 71 | 71 | 562 | 68.47 |
| [runs/detect/predict3](<../runs/detect/predict3>) | 71 | 71 | 562 | 64.19 |
| [runs/detect/predict4](<../runs/detect/predict4>) | 48 | 48 | 374 | 43.73 |
| [runs/detect/predict5](<../runs/detect/predict5>) | 48 | 48 | 374 | 43.73 |
| [runs/detect/predict6](<../runs/detect/predict6>) | 48 | 48 | 374 | 43.73 |
| [runs/detect/predict7](<../runs/detect/predict7>) | 138 | 138 | 1273 | 115.53 |
| [runs/detect/predict8](<../runs/detect/predict8>) | 309 | 309 | 2726 | 265.95 |
| [runs/detect/predict9](<../runs/detect/predict9>) | 197 | 197 | 1786 | 164.82 |
| [runs/detect/predict_largest](<../runs/detect/predict_largest>) | 400 | 400 | 2624 | 335.94 |
| [runs_pose/pred_320](<../runs_pose/pred_320>) | 0 | 0 | 未保存标签 | 0.62 |
| [runs_pose/predict_400](<../runs_pose/predict_400>) | 400 | 400 | 3143 | 345.33 |
| [runs_pose/predict_ice80_test](<../runs_pose/predict_ice80_test>) | 10 | 0 | 未保存标签 | 9.34 |

`pred_320` 目前只有 raw.json，包含 320 项记录；不是 320 张可视化图片目录。predict_400 的 stats.json 记录 400 张、合计 3143 个预测；与 400 个标签文件的行数一致。原说明注明使用 ice80_yolo26/best.pt、imgsz=1536、conf=0.25；本次没有重跑确认。

predict_largest 当前保存 400 张图、2624 条标签；现行 mypredict_largest.py 包含同簇抑制逻辑，仅凭目录名不能断言每图只保留一个目标，也不能反推历史参数。

## 像素级评估（从 summary.json 读取）

同一表块内看同一验证集；不同验证集之间不直接排名。微调实验的 raw_val 记录通常评估 last.pt，原模型评估 best.pt，详见 JSON 的 weight 字段。所有误差为图像像素，不是毫米测量误差。

### 20 张验证图

| 记录 | GT/预测/匹配 | 漏匹配/多余框 | tip 中位 px | tip P90 px | tip 成功率@10%（全部有 tip 的 GT） | tip 出框率 |
|---|---|---|---:|---:|---:|---:|
| [runs_pose/tip_refine/experiments/control_tip1_b1_10ep/raw_val/summary.json](<../runs_pose/tip_refine/experiments/control_tip1_b1_10ep/raw_val/summary.json>) | 132/185/130 | 2/55 | 87.19 | 333.90 | 42.42% | 10.27% |
| [runs_pose/tip_refine/experiments/head_control_tip1_10ep/raw_val/summary.json](<../runs_pose/tip_refine/experiments/head_control_tip1_10ep/raw_val/summary.json>) | 132/156/129 | 3/27 | 67.78 | 313.17 | 47.73% | 22.44% |
| [runs_pose/tip_refine/experiments/head_tip3_10ep/raw_val/summary.json](<../runs_pose/tip_refine/experiments/head_tip3_10ep/raw_val/summary.json>) | 132/156/129 | 3/27 | 67.86 | 309.44 | 49.24% | 19.87% |
| [runs_pose/tip_refine/experiments/smoke_tip3_v2/raw_val/summary.json](<../runs_pose/tip_refine/experiments/smoke_tip3_v2/raw_val/summary.json>) | 132/155/128 | 4/27 | 90.61 | 336.56 | 37.88% | 18.71% |
| [runs_pose/tip_refine/experiments/tip3_b1_10ep/raw_val/summary.json](<../runs_pose/tip_refine/experiments/tip3_b1_10ep/raw_val/summary.json>) | 132/178/130 | 2/48 | 80.47 | 321.65 | 45.45% | 7.87% |
| [runs_pose/tip_refine/original_val/summary.json](<../runs_pose/tip_refine/original_val/summary.json>) | 132/156/129 | 3/27 | 74.51 | 322.61 | 50.00% | 14.10% |

### 38 张验证图

| 记录 | GT/预测/匹配 | 漏匹配/多余框 | tip 中位 px | tip P90 px | tip 成功率@10%（全部有 tip 的 GT） | tip 出框率 |
|---|---|---|---:|---:|---:|---:|
| [runs_pose/tip_refine_232/experiments/control_tip1_232_b1_10ep/raw_val/summary.json](<../runs_pose/tip_refine_232/experiments/control_tip1_232_b1_10ep/raw_val/summary.json>) | 286/294/262 | 24/32 | 67.12 | 172.25 | 55.44% | 31.97% |
| [runs_pose/tip_refine_232/experiments/tip3_232_b1_10ep/raw_val/summary.json](<../runs_pose/tip_refine_232/experiments/tip3_232_b1_10ep/raw_val/summary.json>) | 286/297/262 | 24/35 | 67.74 | 196.58 | 55.79% | 24.24% |
| [runs_pose/tip_refine_232/original_val/summary.json](<../runs_pose/tip_refine_232/original_val/summary.json>) | 286/315/270 | 16/45 | 54.87 | 245.14 | 64.56% | 19.68% |

232 张快照的 38 张 val 上，原模型 tip 中位误差 54.87px、P90 245.14px、全部 GT 成功率 64.56%；等权微调为 67.12px、172.25px、55.44%；tip3 为 67.74px、196.58px、55.79%。微调降低了 P90，但中位误差和全部 GT 成功率变差，不能概括为所有指标都改善或都恶化。tip3 相比等权降低了出框率（31.97%→24.24%），没有改善中位定位误差。

## 复核入口与原始报告

- [独立诊断报告（分辨率、OKS、相邻帧泄漏、规则修正）](<../runs_pose/codex_review_20261022/diagnostic_report.md>)。独立复算的 test 为 10 张，不能与上面的 val 数值混用。
- [Easy80 微调试运行报告与逐图对照](<../runs_pose/tip_refine/pilot_comparison/results_report.md>)。
- [232 张数据的可疑框复检清单、按图汇总和总览图](<../runs_pose/tip_refine_232/annotation_review>)。几何筛选是人工复核线索，不是标注错误的最终判定。

旧说明中的“NMS-free”“直接用框底规则替换 tip”等结论，在独立诊断报告中有复核和限制说明；本索引保留原报告，不把这些旧建议当作已验证默认方案。Pose mAP 也不等于尖端像素精度。

## 本次归类的结果附件

### 标注与检测复核

- [annotation_preview_first16.jpg](<attachments/annotation_detection_review/annotation_preview_first16.jpg>)
- [icicle_detection_review.csv](<attachments/annotation_detection_review/icicle_detection_review.csv>)
- [overlap_review_001_11_17.jpg](<attachments/annotation_detection_review/overlap_review_001_11_17.jpg>)
- [overlap_review_001_17_24.jpg](<attachments/annotation_detection_review/overlap_review_001_17_24.jpg>)
- [overlap_review_001_25_37.jpg](<attachments/annotation_detection_review/overlap_review_001_25_37.jpg>)
- [overlap_review_002_27_35.jpg](<attachments/annotation_detection_review/overlap_review_002_27_35.jpg>)
- [overlap_review_002_37_55.jpg](<attachments/annotation_detection_review/overlap_review_002_37_55.jpg>)
- [overlap_review_002_67_79.jpg](<attachments/annotation_detection_review/overlap_review_002_67_79.jpg>)
- [overlap_review_003.jpg](<attachments/annotation_detection_review/overlap_review_003.jpg>)
- [overlap_review_004.jpg](<attachments/annotation_detection_review/overlap_review_004.jpg>)
- [overlap_review_005_09_34.jpg](<attachments/annotation_detection_review/overlap_review_005_09_34.jpg>)
- [overlay_001_00009.jpg](<attachments/annotation_detection_review/overlay_001_00009.jpg>)
- [review_001_00009-24.jpg](<attachments/annotation_detection_review/review_001_00009-24.jpg>)

### 测距与点云截图

- [edge_cloud_all.png](<attachments/measurement_pointcloud_screenshots/edge_cloud_all.png>)
- [edge_cloud_debug.png](<attachments/measurement_pointcloud_screenshots/edge_cloud_debug.png>)
- [edge_cloud_local.png](<attachments/measurement_pointcloud_screenshots/edge_cloud_local.png>)
- [edge_cloud_metric_perspective.png](<attachments/measurement_pointcloud_screenshots/edge_cloud_metric_perspective.png>)
- [edge_cloud_metric_rotated.png](<attachments/measurement_pointcloud_screenshots/edge_cloud_metric_rotated.png>)
- [edge_cloud_rotated.png](<attachments/measurement_pointcloud_screenshots/edge_cloud_rotated.png>)
- [edge_measurement_full_ui.png](<attachments/measurement_pointcloud_screenshots/edge_measurement_full_ui.png>)
- [pointcloud_aligned_axes.png](<attachments/measurement_pointcloud_screenshots/pointcloud_aligned_axes.png>)
- [pointcloud_before_github_restored.png](<attachments/measurement_pointcloud_screenshots/pointcloud_before_github_restored.png>)

### 图表复核_20261025

- [criteria.png](<attachments/chart_review_20261025/criteria.png>)
- [criteria_1366.png](<attachments/chart_review_20261025/criteria_1366.png>)
- [criteria_1500.png](<attachments/chart_review_20261025/criteria_1500.png>)
- [histogram.png](<attachments/chart_review_20261025/histogram.png>)
- [histogram_1366.png](<attachments/chart_review_20261025/histogram_1366.png>)
- [histogram_1500.png](<attachments/chart_review_20261025/histogram_1500.png>)
- [scatter.png](<attachments/chart_review_20261025/scatter.png>)
- [scatter_1366.png](<attachments/chart_review_20261025/scatter_1366.png>)
- [scatter_1500.png](<attachments/chart_review_20261025/scatter_1500.png>)
- [window_counts.png](<attachments/chart_review_20261025/window_counts.png>)
- [window_counts_1366.png](<attachments/chart_review_20261025/window_counts_1366.png>)
- [window_counts_1500.png](<attachments/chart_review_20261025/window_counts_1500.png>)

### 历史日志

- [predict_run.log](<attachments/logs/predict_run.log>)

检测复核 CSV 为 400 图，GT 合计 2912，预测合计 3423，extra_pred 合计 511。与现存 predict/predict10 的 3414 条标签并不相同，不能直接认定来自这两次预测。avg_iou 是该复核脚本口径，不能代替 AP。

历史日志 predict_run.log 写着 400 张并指向 runs/detect/predict4，但现存 predict4 只有 48 张图和 48 个标签文件。说明日志与当前目录快照无法一一对应，归属待确认；本次没有据此补全或改写历史。测距、点云和图表截图归为界面调试材料，不计入模型精度。
