# 冰凌自动化尺寸测量项目

双击 **启动冰凌测距.cmd** 打开双工作页工作台；**启动测距分析.cmd** 打开原离线/模拟入口。

- [最终成果](final_results/README.md)：当前采用的两点 Pose 模型、400 张预测结果、训练记录与78张数据集。
- [历史归档](history_archive/README.md)：旧检测、三点 Pose、微调实验和早期测试，集中保留。
- `Yolov11/`：保留本地 Ultralytics 源码、原有数据与参考资料；旧实验已迁入历史归档。
- `Measurement/`：模块化RGB-D测距工作台的完整副本。模块说明在 `Measurement/docs/module-map.md`。
- `Measurement/assets/rgb/`：400张原图的独立副本，已逐文件校验。

本机基础解释器仍为 `D:\318`。测距工作台沿用基础环境里的 Ultralytics；`Yolov11/` 中保留的是另一份本地源码。当前两点 Pose 权重为 `final_results/model/best.pt`，伪深度工作页默认读取 `final_results/predictions/labels/`。桌面400张预测结果快捷方式也指向 `final_results/predictions/`。

本次归档清单与验收记录在 `history_archive/migration/`，原迁移资料仍在 `Measurement/docs/migration/`。新运行沿用 `Measurement/training/pose_2kpt/runs/` 和 `Measurement/outputs/ice_pose_runs/`，训练、预测、重绘及预览各自生成时间戳目录，不自动覆盖最终成果。

原始 `D:\yolov11`、桌面原图、Depth Anything、MobileSAM、伪深度缓存及自动关机脚本均保留原位。本次移动的是本项目中的副本。历史参数、日志及归档脚本中的旧绝对路径保留作来源证据，不能直接作为当前重训入口。

可选模型依赖仍共享本机Python，不是无需外部环境的便携包。
