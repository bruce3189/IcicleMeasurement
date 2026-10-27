# 附录 D · 回归病历与测试矩阵

> 两个用途：改代码前知道"哪些检查能兜住你"，改坏之后知道"这个症状以前出现过"。
> 来源：`D:\ipa_work\pointcloud_verify\修复说明.md`、`repro_coords_output.txt`、代码注释、`tests/` 实读。

## 1. 一句话

**这个项目的测试覆盖不算薄（101 个用例 + 24 项几何 + 58 项压力），但六次重大 bug 里有三次是测试完全没报错的。** 下面每条的"防护"列才是重点。

## 2. 六次重大 bug 病历

### 病历 1：Z 坐标失真 200 倍（最严重）

**症状**：点云看起来是平的，深度方向几乎没有起伏。UI 上端点显示 `Z = −1.028`，但实际只有 `−0.00525 m`。

**根因**：`set_data` 里存在**两套互不相干的比例**：
- X/Y 走反投影后乘 `xy_display_scale`（16.052 /米）
- Z 走「深度簇基线压缩 + 残差放大」（基线 3.995、残差系数 201.619）

两者量纲完全不同，Z 被放大了约 200 倍。因为点云"看起来还有立体感"，肉眼和测试都没发现。

**定位方法**：`repro_coords.py` 把修复前（`legacy_*`）与修复后（`build_cloud`）的坐标并列复算，同一像素两套结果对照。

**修复**：全部改为 `geometry.build_cloud` 的**单一等比系数**（`isotropic_scale` 取 `radius / max(X/Y/Z 轴跨度)`，XYZ 共用）。边缘样例 `simulator.generate('edge', seed=42)`、端点 (5.0, 200.0)、radius 8、238 个有效像素、目标簇 99 的实测对照：

| | X | Y | Z |
|---|---:|---:|---:|
| 修复前显示坐标 | 0.0407 | −0.0212 | **−1.0280** |
| 真实米制相对坐标 | 0.002535 | −0.001321 | **−0.005250** |

`|Δ| = 0.005977 m`，等比系数 8.7382。

**防护**：`test_pointcloud_geometry.py` 的「偏移等于反投影差」8 项，包括"原始深度未被吸附到原点"（sample 1.494130 vs origin 1.499379）、"端点 Z 偏移是毫米级"（−0.005250 m）。

---

### 病历 2：类别信息污染几何

**症状**：把某个点的类别改一下（只应该是换个颜色），测出的长度会变。

**根因**：两处隐式耦合
1. 抽稀按类别配额取样，类别变了抽到的点就变了
2. `focus_radius_m` 取"目标簇距离的 98 分位"——这个值直接依赖聚类标签

**修复**：抽稀改 `np.linspace` 均匀抽样（去掉类别依赖）；`focus_radius_m` 改由搜索窗在参考深度下的半对角线 ×1.6、下限 0.03 m 计算，**不依赖任何标签**。类别只写进 `vertex_categories` 用于着色，`build_cloud` 的入参里根本没有类别。

**防护**：`test_pointcloud_geometry.py` 的「类别不动点」2 项——改类别后 `vertices` 与 `relative_m` 逐字节相等，最大差异 `0.000e+00 m`。

---

### 病历 3：点云整块消失且不报错

**症状**：打开点云页，什么都没有。控制台无报错，或只有一个 `TypeError: object of type 'bool' has no len()`。

**根因**：`paintEvent` 里这行

```python
for visible, tip, color, label in axis_specs:
```

把外层的 numpy 布尔掩膜 `visible` 覆盖成了单个 `bool`。后面 `len(visible)` 直接抛异常，整个绘制中断。

**修复**：循环变量改名（现代码 `for show, tip, color, name in (...)`）。

**防护**：无自动化测试覆盖。**这条只能靠命名规范防**——绘制循环里的变量名必须与外层隔离。

---

### 病历 4：稳定性分数虚高（假阳性通过闸门）

**症状**：某些端点在证据明显不足时被 `accepted`。

**根因**：窗口半径被框短边截断后，继续增大半径时**窗口实际范围没变**，但每次都记一次"连续通过"，凭空凑出 `consecutive ≥ 2`。

**修复**：`core.py` 里记 `bounds == previous_bounds` 时不重复计分；到上限仍不过就 `break` 立即拒绝。

**防护**：`test_window_evidence.py` 的「密集半径步长不能制造连续通过」用例。

---

### 病历 5：透视模式下点跑到相机后方

**症状**：切到透视投影后，点云出现镜像鬼影。

**根因**：相机被放在视线后方 `6×radius` 处，当顶点视深度 ≥ 该距离时，`depth` 变成负数，投影穿过相机形成镜像。

**修复**：绘制前剔除 `~isfinite(screen)` 与 `view_depth ≥ distance` 的点，`depth` 下限 1e-3。

**防护**：`test_cloud_navigation.py` 的旋转/缩放后几何不变检查。

---

### 病历 6：离屏渲染崩溃

**症状**：无窗口后端（offscreen）下跑渲染脚本，`QWidget.grab()` 直接让进程退出（exit=127，无输出）。

**根因**：`grab()` 依赖窗口后端；沙箱化/离屏环境里没有。

**修复**：改 `QImage` + `QPainter.render()` 直绘。离屏环境还需手动 `addApplicationFont("msyh.ttc")`，否则中文全是方框。

**防护**：`render_check.py` 自成一体，不依赖窗口系统。

---

### 另外三个非几何坑

| 坑 | 症状 | 处理 |
|---|---|---|
| 测试过期（非回归） | `test_gui` 硬编码 `tabText(0)=='逐样本分布'`；`test_measure.edge` 期望 accepted，与"短边上限即拒绝"设计冲突 | 改按标题断言；edge 期望改 `rejected` 并校验原因含"短边搜索上限" |
| 采集帧错位 | 预览帧与深度/检测不同步 | 单槽邮箱丢过期帧；worker 持有自己的 packet |
| 裁剪窗口重复 | 见病历 4 | 窗口去重后再评分，**去重逻辑视为不可删** |

## 3. 测试矩阵

### 3.1 unittest（`python -m unittest discover -s tests`）

12 个文件，**101 个用例**：

| 文件 | 用例 | 覆盖什么 |
|---|---:|---|
| `test_measure.py` | 19 | 无噪声几何、毫米量化、算术均值口径、空洞/误读背景、噪声、全部拒绝条件、成员变化、边界裁剪、双端点联合门控、无效输入、"取最小值而非平均值"、30 种子噪声基准、edge 场景按设计拒绝 |
| `test_pseudo_depth.py` | 14 | 伪深度场景构造、Z16 导出、工程保存/重载 |
| `test_real_input.py` | 11 | 16bit PNG 毫米深度、输入包往返、`mm/raw` 单位与元数据错误、YOLO 可见性与顺序、原始比例重放、采集单槽邮箱、SDK 对齐/设备尺度/生命周期、校正保边、超时与停止释放、Pose 实例顺序与 BGR 转换 |
| `test_window_evidence.py` | 10 | 锚区选本地分量而非更远更大分量、平缓高斯不应被二次分簇判成双表面、被重分类的旧像素不算新证据、共有像素 Jaccard 满分仍可因新增区域失败、密集步长不能制造连续通过、平面秩与覆盖失败可追溯 |
| `test_unified_workspace.py` | 10 | 双标签页结构、输入版本号调度、任务取消与重算、配置往返 |
| `test_keyboard_smooth_zoom.py` | 8 | 缩放 0.05–80 全程鼠标锚定、Shift 精调、水平滚轮忽略、逐帧锚点、快速反向滚轮不追旧目标 |
| `test_pseudo_refinement.py` | 7 | 伪深度表面精修、背景平面约束 |
| `test_cloud_navigation.py` | 6 | 聚焦样本 360° 旋转与缩放后仍在中心、返回端点/复位不改几何、适配只看点云不看坐标轴、内联展开保留画布状态、WASD 方向与松键停止、文本框输入 WASD 不动点云 |
| `test_pixel_inspector.py` | 6 | 鼠标位置读取真实像素/深度，不从预览色反推 |
| `test_gui.py` | 4 | 导入→计算→选点→改参→导出全链路、真实线程取消、worker 异常释放 UI、标签页结构 |
| `test_near_mean_zoom.py` | 4 | 近簇均值模式下的缩放行为 |
| `test_ui_module_imports.py` | 2 | 模块导入不拉起重依赖（主进程不导入 torch） |

### 3.2 脚本型检查（不在 discover 里，要手动跑）

| 脚本 | 项目数 | 内容 |
|---|---:|---|
| `icicle_measure/test_pointcloud_geometry.py` | 24 | 分四组：①拒绝不被放宽（4）②偏移等于反投影差（8）③类别不动点（2）④旋转与投影（5） |
| `icicle_measure/stress_test.py` | 20 节 / 58 项 | 见下 |

**stress_test 的 20 个维度**：

1. 同种子重复确定性（10 种子 × 2 次）
2. 噪声扫描 1→50 mm
3. 深度恰在 min/max 边界
4. 全均匀深度（Otsu 无法分簇）
5. 极端内参（fx=10000 / 1.0 / 0 抛异常）
6. 检测框四角与越界
7. 重复 ID 抛异常
8. `consecutive=1/5` 敏感性
9. `min_samples=1/10000`
10. 离群比例 1%→40%
11. 七个拒绝场景的拒绝原因关键词正确性
12. 多目标准确度（2 accepted + 1 skipped）
13. 厘米级/微米级量化
14. 50 目标性能（<5 s）
15. Otsu 边界（微小差可分开、零差不分、单像素）
16. `confidence_scores` 边界（满分 1.0 / 最差 0）
17. NaN/Inf 深度
18. 全零深度必拒
19. 误差随噪声单调不突变
20. `Config` 五项非法组合必抛 `ValueError`

### 3.3 benchmark

```bash
python -m icicle_measure.benchmark --output outputs/rgbd_validation
```

- 30 个固定种子 `noisy` 场景，`noise_sigma_mm = 3`
- 另跑 8 个必拒场景（`sparse/overlap/background/empty/drift/membership/interference/touching_surfaces`）
- 通过条件：accepted ≥ 27/30、端点误差 ≤10mm、长度误差 ≤20mm、8 个拒绝检查全过
- **当前实测**：30/30、端点最大误差 **0.6165 mm**、长度最大误差 0.3341 mm

⚠ `outputs/rgbd_validation/benchmark.json` 是旧版产物（缺 `slope_mode_comparison` 字段），要引用前先重跑覆盖。

## 4. 改动类型 → 必跑检查

| 改了什么 | 必跑 |
|---|---|
| `geometry.py` / 点云坐标 / 投影 | `test_pointcloud_geometry.py` + `test_cloud_navigation.py` + benchmark |
| `core.py` / `evidence.py` / 任何阈值 | 全量 `unittest discover` + `stress_test.py` + benchmark |
| `gui.py` / `ui/*` | `unittest discover`（含 `test_gui` `test_unified_workspace` `test_keyboard_smooth_zoom`） |
| `pseudo_*` | `test_pseudo_depth` + `test_pseudo_refinement` + `test_ui_module_imports` |
| 标签解析 / `io.py` / 数据格式 | `test_real_input` + `test_measure` |

**改前先跑一遍建立基线**，否则改完分不清是新增失败还是原来就有。

## 5. 测试抓不住的三件事

这三条必须靠规范和代码审查：

1. **绘制循环里复用外层变量名** → 点云静默消失（病历 3）
2. **`dataset.yaml` 路径失效** → 训练到空数据，指标"变差"的结论完全是假的
3. **类别重新参与几何计算** → 坐标随标签漂移，测试不报警（病历 2 的回归）

## 6. 性能基线

| 场景 | 耗时 |
|---|---|
| 238 点单帧 | 2.5 ms |
| 10000 点单帧 | 约 6 ms |
| 50 个目标整帧 | < 5 s |
| `mouseMoveEvent` 优化后 | 较原线性扫描快 415× |

如果改动后发现点云拖拽明显变卡，先查是否退回了逐点 Python 循环——当前实现是按类别分组的 `QPolygonF` + 一次 `drawPoints` 批量调用。

## 7. 环境四坑

1. **三套 requirements 用途不同**，别混装：
   - `requirements.txt` —— 主依赖（numpy / opencv-python / PyQt5）
   - `requirements-camera-py39.txt` —— 只有 pyrealsense2，且**限 Python 3.9**（官方 wrapper 对 3.9 只支持 ≤2.57.4 的发布）
   - `requirements-pseudo-depth.txt` —— timm / huggingface-hub / safetensors，要装在 `work/pseudo_env`
2. **伪深度必须用独立 venv**（`work/pseudo_env`），用 `setup_pseudo_depth.py` 安装。主进程不导入 torch/timm 是刻意设计，别为了省事把依赖并进主环境。
3. **RealSense 只能在 py3.9 下用**，主环境若是 3.10+ 就用 `requirements-camera-py39.txt` 那一套。
4. **两个 launch 脚本是两条不同入口**：`launch_measurement.cmd`（主链路）与 `launch_pseudo_depth.cmd`（伪深度）。不是同一程序的开关。

## 8. 改码约定

- **改前先跑基线**：`python -m unittest discover -s tests`（101 用例）。不建基线就分不清"改坏了"和"本来就红"。
- **一次只改一层**：界面层（`gui.py` / `ui/`）、计算层（`core` / `evidence` / `geometry`）、基础层（`models` / `io` / `render` / `simulator` / `capture` / `pose`）。跨层改动会让失败原因无法定位。
- **不新建与 `docs/` 重复的说明文档**：要补充就改现有那篇。
- **不删 `outputs/` 下的历史快照**：那是每次验证的证据，只增不改。
- **新脚本归入既有目录**（`work/` 放实验性，`icicle_measure/` 放正式模块），不往仓库根堆。
- **提交信息写「做了什么 + 为什么」**：决策理由比 diff 更容易丢，而它是半年后唯一能追溯的东西。

### 内存紧张时的替代跑法

```bash
python work/check_tests_grouped.py      # 逐文件在独立进程执行同一批测试，不改断言
```

