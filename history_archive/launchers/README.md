# 旧启动入口归档

当前使用项目根目录的 [launch_workbench.cmd](../../launch_workbench.cmd)，默认进入场景生成，在同一窗口切换测距分析。

- root_scene.cmd：原根目录“启动冰凌测距.cmd”，完整搬入。
- root_analysis.cmd：原根目录“启动测距分析.cmd”，完整搬入。
- measurement_scene.cmd：原 Measurement/launch_pseudo_depth.cmd 的内容快照。
- measurement_analysis.cmd：原 Measurement/launch_measurement.cmd 的内容快照。

四份原件均按字节保留，哈希与原路径记录在 manifest.json。归档脚本里的相对路径属于原位置，仅用于追溯，不作为当前启动入口。Measurement 中的两个兼容脚本均已转发到统一入口。
