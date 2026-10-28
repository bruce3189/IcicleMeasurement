# 重点范例：005_00036

用户指定的重点范例。后续工作台演示、功能说明与论文配图优先采用这张图的两点预测和对应模拟深度。此处固定保存现有输出，供查阅、重载和取图。

## 优先使用的图片与数据

| 用途 | 文件 |
|---|---|
| 两点预测，细线版 | [prediction_thin.jpg](prediction_thin.jpg) |
| 两点预测，原输出 | [prediction.jpg](prediction.jpg) |
| 模拟观测深度预览，含空洞与误差 | [preview_observed.png](pseudo_depth/preview_observed.png) |
| 完整模拟表面深度预览，对照用 | [preview_complete.png](pseudo_depth/preview_complete.png) |
| 深度色标，0.5–4 m | [depth_colorbar_m.png](pseudo_depth/depth_colorbar_m.png) |
| 预测、掩膜、背景与深度六宫格 | [overview.jpg](pseudo_depth/overview.jpg) |
| 观测深度数值，uint16 PNG | [depth_observed_z16.png](pseudo_depth/depth_observed_z16.png) |
| 完整深度数值，uint16 PNG | [depth_complete_z16.png](pseudo_depth/depth_complete_z16.png) |
| 对应原图与两点标签 | [original_rgb.jpg](pseudo_depth/original_rgb.jpg)、[original_prediction.txt](pseudo_depth/original_prediction.txt) |

数值深度的0表示无效，其他值乘0.001得到Z深度，单位为米。彩色预览仅用于显示，不能代替数值深度输入。

![005_00036 预测与模拟深度总览](pseudo_depth/overview.jpg)

## 配对与使用说明

已逐字节核实：模拟深度项目保存的原图与 `Measurement/assets/rgb/005_00036.jpg` 一致；原预测标签与最终预测目录中同名标签一致，共5个预测实例。预测图片保留原图分辨率1644×3840，深度为等比留边的1280×720；坐标变换记录在项目参数中。

- 测距输入：在工作台“测距分析”中导入 [input_observed.json](pseudo_depth/input_observed.json)；完整表面对照使用 [input_complete.json](pseudo_depth/input_complete.json)。
- 编辑或重载场景：在“场景生成”中加载 [pseudo_project.json](pseudo_depth/pseudo_project.json)。它的原图和标签路径已指向包内副本；其他历史报告的原路径保留作来源证据。
- 保存的测距分析：[观测深度结果](pseudo_depth/measurement_observed/measurement.json)、[完整深度结果](pseudo_depth/measurement_complete/measurement.json)。观测深度中3个目标通过、2个拒绝；完整深度中2个通过、3个拒绝，拒绝项及原因均保留。
- 原始工作台输出仍在 `Measurement/outputs/pseudo_depth_samples/005_00036/`。本范例是44个既有产物的固定副本，文件来源及哈希见manifest.json。

本例展示的是“模型伪深度 / 人工合成”流程，使用假设尺度与虚拟内参。它可说明预测、深度和测距判定的对应关系，不能作为真实D435i深度或真实冰凌毫米精度的验证结果。
