# 冰凌自动化尺寸测量项目

双击 **启动冰凌测距.cmd** 打开双工作页工作台；**启动测距分析.cmd** 打开原离线/模拟入口。

- `Yolov11/`：原D盘检测/训练工程的完整副本，保留本地Ultralytics源码及历史实验。
- [YOLO 历史训练与结果索引](Yolov11/历史训练与结果/README.md)：训练参数与指标、权重、预测输出和复核材料的统一入口。
- `Measurement/`：模块化RGB-D测距工作台的完整副本。模块说明在 `Measurement/docs/module-map.md`。
- `Measurement/assets/rgb/`：400张原图的独立副本，已逐文件校验。

本机基础解释器仍为 `D:\318`。测距工作台沿用基础环境里的Ultralytics；从Yolov11目录运行则沿用该目录的本地Ultralytics，二者不混用。当前两点Pose权重仍在 `Measurement/training/pose_2kpt/runs/yolo11n_pose_78_2kpt/weights/best.pt`；历史检测权重没有替换。

迁移清单、外部依赖、已有缺失项与验收结果在 `Measurement/docs/migration/`。2026-10-28 核查确认：`iceice.yaml` 位于 `Yolov11/ultralytics/cfg/datasets/`，`mytrain.py` 使用相对名；根目录没有同名文件。本次没有替换数据集或启动训练。

旧目录和桌面原图全部保留。2026-10-28 已整理 YOLO 历史记录：训练和预测目录保留原路径，根目录散落的复核图、截图、日志与旧图表复核目录中的文件归入 `Yolov11/历史训练与结果/结果附件/`，原路径与哈希见该目录的 `整理清单.json`。旧日志、备份和权重内记录的训练路径属于历史来源；部分旧训练脚本仍含旧目录绝对路径，不能直接作为已适配的重训入口。

可选模型依赖仍共享本机Python，不是无需外部环境的便携包。
