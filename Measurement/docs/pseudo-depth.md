# RGB 场景伪深度与 D435i Z16 仿真

主界面右上角 **RGB 生成伪深度** 打开编辑窗口。当前开放单张和已批准的六张样例，未处理整批 400 张。

## 查看与修改

重点范例为 [005_00036](../../final_results/examples/005_00036/README.md)，后续演示和图文说明优先使用其两点预测与对应模拟深度。场景生成下拉框选择 `005_00036` 可查看原有项目；固定范例包还保存完整/带误差深度、两种预测图、色标、项目参数及测距结果。

总览位于 `outputs/pseudo_depth_samples/review.html`，六张联系图为同目录的 `six_samples.jpg`。生成窗口的样例下拉框会自动加载已有项目。点击“送入主界面测距 / 点云”，查看实际扩窗判定、端点图表和交互点云；默认送测带误差观测，也可切换完整表面。

1. 选择目标实例。查看模式支持滚轮缩放、拖动平移和双击适配。
2. 添加正/负提示点，再点击“按正负点重新分割当前目标”。其他实例的人工掩膜保留，YOLO 关键点不会自动作为分割提示。
3. 用画笔增加或擦除掩膜，点击“应用修正”重建深度。
4. 取消“使用自动基线”后设置目标 Z；轴向首尾差改变表面趋势，圆钝横截面的半径来自掩膜宽度及假设几何。
5. 背景画笔设置所画区域的深度偏移，避开目标；偏移设零可清除修改。
6. 在“尺度与缺陷”修改映射、空洞、背景回波、边界错误、噪声和种子，再应用修正。
7. 保存项目后，可重载模型层、修正掩膜、背景偏移、提示点、数值参数与动作记录，无须重新推理。

待应用修改不能直接保存或送测。更换 RGB/标签须重新生成，避免旧深度与新图片配对。目标参数按实例保存，切换实例不会丢失已输入的数值。

## 输出

| 文件 | 含义 |
|---|---|
| `original_rgb.*` / `original_prediction.*` | 原始 RGB、原预测标签副本 |
| `rgb.png` / `detections.json` | 1280×720 等比留边图与同步变换的框、原关键点 |
| `depth_complete_z16.png` / `depth_observed_z16.png` | 单通道 uint16，0 无效，乘清单中比例得到 Z 米值 |
| `depth_complete_m.npy` / `depth_observed_m.npy` | 与 PNG 量化值对应的米制 float32 |
| `latent_surface_m.npy` | 量化前假想表面，可检查量化误差 |
| `model_relative.npy` / `background_m.npy` | 模型原始相对深度与假设米制映射 |
| `instances.png` | 顺序实例索引图；原实例 ID 保留在结构化检测中 |
| `valid_region.png` / `valid_depth_*.png` | 原图内容区域、两份深度各自的有效像素 |
| `source_map.png` / `uncertainty.png` | 来源类别与逐像素诊断位标志；不是正确概率，图例见元数据 |
| `input.json` / `input_observed.json` / `input_complete.json` | 可直接导入测距界面的数据清单 |
| `pseudo_project.json` / `editable_layers.npz` | 配置、版本、人工修改及可重载原始层 |
| `measurement_observed/` / `measurement_complete/` | 两种输入的实际测距 JSON、CSV、标注图 |
| `generation_report.json` / `timing_and_measurement.json` | 逐目标假设、问题、判定与耗时 |

所有米制预览固定色标 0.5–4 m，超范围只截色、不改存储值。模型原始相对深度单独显示其 P2/P98 归一化范围。相对深度按本图记录的参数映射到人工设定尺度，不构成跨照片绝对距离标定。

## 安装与运行

模型和环境位于 `work/pseudo_models`、`work/pseudo_env`；`models.lock.json` 记录官方代码版本、权重 SHA256 和依赖版本。推理前校验权重，缓存键包含 RGB、标签、模型锁、尺寸、提示点及适配器版本。主界面不导入 torch/timm，模型在可取消的独立子进程执行。

```powershell
# 按本机实际代理填写；无代理时省略 --proxy。
D:\318\python.exe setup_pseudo_depth.py --proxy http://127.0.0.1:7897
# 只生成六张已批准样例。
D:\318\python.exe -m icicle_measure.pseudo_samples
# 仅更新总览。
D:\318\python.exe -m icicle_measure.pseudo_samples --review-only
```

独立环境共享现有 CPU PyTorch、torchvision、NumPy、OpenCV、SciPy、Pillow 和 PyQt；新增固定版本依赖安装在其中，不升级主环境。下载失败会明确报错，不以随机或固定背景代替模型。已有可编辑项目可离线查看、修正与合成。

## 解释与限制

- 仿真边界是**对齐到彩色帧后的 Z16 接口**。虚拟内参由名义彩色视场角 69°×42°构造，非 D435i 工厂标定；可导入原 RGB 分辨率的 `fx/fy/cx/cy`，随缩放同步变换。
- 原照片没有标定资料。保存 `undistorted=false` 和 `geometry_mode=assumed_pinhole`，仅明确合成来源允许该假设；真实输入仍须对齐、去畸变。
- 0.5–2 m 是目标基线先验；缺乏模型前景支持时使用显式 1.25 m 假设。表面半径和轴向趋势可能使局部 Z 略超出基线范围。
- 长度常见范围为 18–22 cm、极限参考范围为 17–25 cm。落在极限内但超出常见范围时单独提示；超出极限时报告异常。两者均不参与深度拟合或测量长度修正。由先验生成的数据不能证明真实长度精度。
- 检测框约束候选位置；近乎矩形填满或主要在框外的掩膜拒绝。重叠争议像素无效，待人工修正。
- 空洞、背景回波、边界错误及距离相关噪声为工程假设，不是 D435i 实测误差分布。种子可重现，缺陷可逐项关闭。
- 单张照片未恢复真实双目视场、外参、投射器、曝光及透明折射误差。该模块不伪造真实设备序列号、IR 帧或 IMU 数据。
- 测距保留原关键点射线与现有全部门槛，失败保留原因。界面与导出均标记“模型伪深度 / 人工合成”和“基于伪深度的估计距离”。

官方依据：[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)、[RealSense 投影与 Z16](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)。上游源码及许可证保存在模型目录，版本和权重校验随项目导出。
