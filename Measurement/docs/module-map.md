# 项目模块与维护入口

本轮仅把既有界面定义迁到独立文件。界面文字、布局、参数、信号连接、工作线程、点云交互、算法与文件格式未重写。`gui.py` 和 `pseudo_gui.py` 继续提供原来的导入路径；它们导出的控件与新模块中的控件是同一个类，不是新增包装控件。

## 入口与工作流

| 文件 | 职责 |
| --- | --- |
| `icicle_measure/__main__.py` | 原命令行启动入口，`python -m icicle_measure`、`--pseudo-depth` 保持兼容 |
| `icicle_measure/gui.py` | 主窗口、双工作页、输入版本与最新任务调度、测距工作线程及旧控件导入入口 |
| `icicle_measure/pseudo_gui.py` | 伪深度界面的旧导入入口，转出下方两个伪深度界面模块的既有定义 |

`Worker` 和 `PoseMeasurementWorker` 保留在 `gui.py`，继续使用同一模块的 `measure_frame`。已有扩展和测试的函数替换入口不变。相机窗口仍在 `live_gui.py`，通过原有按需导入路径打开。

## 界面组件

| 新文件 | 维护内容 |
| --- | --- |
| `ui/common.py` | 字体配置、RGB/深度图像视图、算法参数对话框、共用状态文字 |
| `ui/charts.py` | 深度簇分布、深度与均值变化、扩窗样本量图 |
| `ui/cloud_canvas.py` | 点云坐标、透视绘制、选点、WASD、平滑滚轮、旋转与聚焦 |
| `ui/cloud_panel.py` | 点云按钮、显示选项、样本读数与展开请求 |
| `ui/diagnostics.py` | 测距判定表、可换位的诊断标签页 |
| `ui/pseudo_preview.py` | 伪深度预览图、掩膜画笔视图、鼠标像素定位 |
| `ui/pseudo_workspace.py` | 伪深度工作页、既有后台任务、导入/编辑/保存/自动送测流程与旧独立窗口兼容类 |

画布类保留完整定义，没有拆成逐帧跨模块转发或动态插件。原先的 `from icicle_measure.gui import DepthTerrainCanvas`、`from icicle_measure.pseudo_gui import PseudoDepthPanel` 等写法继续有效；调试器显示的类定义文件会指向对应新模块。

## 原有计算与输入模块

- `models.py`：输入、配置、每窗口诊断与结果类型。
- `core.py`、`evidence.py`：两端点扩窗、分离/样本/稳定性判定、算术均值和测距。
- `geometry.py`、`render.py`：统一反投影与点云投影、RGB/深度标注与欧氏距离文字。
- `io.py`：RGB-D/标签导入、单位校验、输入包与结果导出。
- `capture.py`、`pose.py`、`live_gui.py`：可选相机采集、两点检测、实时工作流。
- `pseudo_depth.py`、`pseudo_geometry.py`：伪深度数据、表面与缺陷合成、背景平面约束、项目保存重载。
- `pseudo_models.py`：独立进程中的可选模型推理与缓存；仍不在主界面进程导入Torch。
- `pixel_inspector.py`：鼠标像素读数；`simulator.py`：固定种子模拟输入。

这些计算和输入模块在本轮保持原文件内容。不要为了改界面显示去改计算层阈值，或为了维护类的位置改变旧启动命令和输入格式。

## 验证与性能

- `work/check_modularization.py before/after` 记录类/函数代码体、18种模拟结果、两张伪深度样例、11张原生Qt截图与耗时。
- `outputs/modularization_review/comparison.json` 保存本轮前后对比；`before/`、`after/` 保存原始报告与截图。
- 常规回归：`python -m unittest discover -s tests`。若机器内存紧张，可用 `python work/check_tests_grouped.py`，在独立进程中逐文件执行同一批测试，不修改程序行为或测试断言。
- 备份：`work/backups/modularization-before-20261026.zip`。`work/extract_ui_modules.py` 是一次性机械迁移记录，已迁移的工程不应再次执行。
