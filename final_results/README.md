# 当前采用的最终模型与预测结果

用户确认采用：YOLO11n 两点 Pose（tip 尖端、base 根部中点），78张数据训练的 best.pt。目录集中日期：2026-10-28T18:52:29+08:00。

**重点范例：[005_00036 的两点预测与模拟深度](examples/005_00036/README.md)**。后续演示和图文说明优先使用这一组，已保存对应预测图、完整/带误差深度、项目参数和测距记录。

- [最终模型](model/best.pt)：唯一选定的检测权重，SHA-256 `ccd5cb349e5ae6f7986d304a1f153982500e9bd4e9d4613c4deb3cc418319bad`。
- [400张预测结果](predictions/)：400张带框和两点的预测图、400份 labels、400张 thin_overlay、两张预览总图以及两份 CSV，共1204个文件。
- [细线版预览总图](predictions/preview_contact_sheet_thin.jpg)；[普通预览总图](predictions/preview_contact_sheet.jpg)。
- [逐图统计](predictions/image_summary.csv)：400条；[关键点坐标与置信度](predictions/keypoints.csv)：2975条实例记录。
- [训练记录](training/run/)：原参数、results.csv、曲线和训练/验证图；[训练数据集](training/dataset/dataset.yaml)：train 57 / val 16 / test 5。

桌面测距工作台默认使用这里的模型和预测标签。当前模型用于两点检测；伪深度的 Depth Anything 和 MobileSAM 仍在 Measurement/work/，不属于旧检测模型。

训练记录中的旧绝对路径保留作来源证据。未选用的 last.pt 与 quick_test 快照在 [早期测试归档](../history_archive/pose_2kpt_early/)。旧检测和三点Pose实验见 [历史归档](../history_archive/README.md)。

现用训练脚本、预测脚本和看板仍位于 Measurement/ 中。后续训练、预测、重绘和预览输出使用新的时间戳目录；本次没有重新训练或重跑400张预测。Git不保存这里的模型、图片或标签，需保留本地文件。
