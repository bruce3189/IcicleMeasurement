# 历史实验归档

这里集中保存用户未选作最终版的检测、三点 Pose、微调、预测和早期测试。当前版本请使用 [final_results](../final_results/README.md)。

- [YOLO旧实验](yolov11_legacy/)：原 runs、runs_pose、weights，以及旧训练/预测入口脚本和初始权重。保留实验内部结构及原始配置、日志内容；旧脚本未作为现用入口适配，不能直接用来重训。
- [历史索引](yolov11_legacy/history_index/README.md)：23项独立历史记录、权重清单、预测与误差复核资料。
- [两点模型早期测试](pose_2kpt_early/)：未选用的 last.pt、quick_test 快照和测试输出，以及空的早期预测尝试目录。
- [旧启动入口](launchers/README.md)：原来的两个根目录启动文件及内部启动脚本原件，现用统一入口为根目录的 launch_workbench.cmd。
- [搬迁与验收清单](migration/README.md)：原路径、英文目标路径、字节数与SHA-256；包括文档引用更新前的原件。

原有两处232张实验目录是Windows Junction，本次将其展开为内容完整的独立目录，保留两处历史记录。没有删除重复权重或预测结果。归档范围仅为本项目副本，D:/yolov11原工程与桌面原图保留。
