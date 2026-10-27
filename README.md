# 冰凌自动化尺寸测量项目

双击 **启动冰凌测距.cmd** 打开双工作页工作台；**启动测距分析.cmd** 打开原离线/模拟入口。

- `Yolov11/`：原D盘检测/训练工程的完整副本，保留本地Ultralytics源码及历史实验。
- `Measurement/`：模块化RGB-D测距工作台的完整副本。模块说明在 `Measurement/docs/module-map.md`。
- `Measurement/assets/rgb/`：400张原图的独立副本，已逐文件校验。

本机基础解释器仍为 `D:\318`。测距工作台沿用基础环境里的Ultralytics；从Yolov11目录运行则沿用该目录的本地Ultralytics，二者不混用。当前两点Pose权重仍在 `Measurement/outputs/ice_pose_runs/yolo11n_pose_78_2kpt/weights/best.pt`；历史检测权重没有替换。

迁移清单、外部依赖、已有缺失项与验收结果在 `Measurement/docs/migration/`。`Yolov11/mytrain.py` 引用的 `iceice.yaml` 原先已缺失，本轮未替换数据集，也未启动训练。

旧目录和桌面原图全部保留。本阶段只归并与适配路径，没有进行内部清理或重排；下一阶段整理结构需另行确认。旧日志、备份和权重内记录的训练路径属于历史来源，不表示新程序会回写旧项目。

可选模型依赖仍共享本机Python，不是无需外部环境的便携包。
