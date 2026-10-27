# 数据集处理 GUI 实施计划

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** 在 `D:\make_dataset` 中实现模块化中文 PyQt5 工具，支持独立项目管理、视频抽帧、两种数据集划分，并为标注和训练预留未接入入口。

**Architecture:** 将可复用逻辑放进不依赖界面的 core/services 层；现有命令行脚本改为薄入口复用服务；GUI 通过 QThread worker 调用服务并接收进度、取消、完成与错误信号。每次操作使用新的运行目录，且所有输出路径必须为空。

**Tech Stack:** Python 3.9、PyQt5、OpenCV、标准库 unittest、JSON 项目元数据。

---

## 实施约束

- 保留现有 `images/`、`labels/`、`iceice/`、`xz_dataset/` 内容和目录结构，不在其上运行处理操作。
- 不移动、删除、改写或合并图片、标签和视频；服务只复制数据。
- 原始视频只在项目 JSON 中记录绝对路径。
- 抽帧、带标签划分、图片抽样都使用唯一运行目录；取消时保留已生成文件并记录取消状态。
- 标注与训练只显示“未接入”入口和模块接口，不实现标注、推理或训练。
- 本仓库当前没有 Git 元数据，实施计划不依赖提交操作。
- 测试和手动验收只使用临时目录和测试视频，不触碰现有数据集。

## 文件布局

新增：

- `app.py`：应用启动入口。
- `dataset_tool/__init__.py`：包声明。
- `dataset_tool/core/models.py`：项目、视频、运行记录的数据模型及状态值。
- `dataset_tool/core/project_store.py`：项目目录创建、路径管理、JSON 读写和重开时源视频可用性检查。
- `dataset_tool/services/video_extractor.py`：可复用的视频抽帧服务。
- `dataset_tool/services/labeled_splitter.py`：图片/标签配对与 70/20/10 复制划分服务。
- `dataset_tool/services/image_sampler.py`：按 20% 抽样并以 70/30 复制到 train/val 的服务。
- `dataset_tool/workers/operation_worker.py`：QThread worker、信号和取消标记。
- `dataset_tool/modules/base.py`、`dataset_tool/modules/registry.py`：模块接口及模块注册表。
- `dataset_tool/ui/main_window.py`：主窗口、页面导航和全局状态提示。
- `dataset_tool/ui/pages/video_page.py`：项目/视频选择、视频预览、抽帧设置、进度、取消及缩略图。
- `dataset_tool/ui/pages/labeled_split_page.py`：带标签数据划分页。
- `dataset_tool/ui/pages/image_sample_page.py`：图片抽样划分页。
- `dataset_tool/ui/pages/unavailable_page.py`：标注和训练占位页。
- `dataset_tool/ui/widgets/thumbnail_gallery.py`：缩略图显示、选中预览和打开文件位置。
- `tests/test_project_store.py`、`tests/test_video_extractor.py`、`tests/test_dataset_services.py`、`tests/test_cli_wrappers.py`：临时数据测试。
- `README.md)：中文安装、启动、操作说明及目录约定。

修改：

- `extract.py`、`split.py`、`make_xz.py)：保留原参数和命令行使用方式，改为解析参数并调用共用服务；错误返回非零退出码。没有参数时仍沿用各自原有默认路径和行为约定。

---

### 阶段 1：建立项目模型和持久化

**文件：** `dataset_tool/core/models.py`、`dataset_tool/core/project_store.py`、`tests/test_project_store.py`

1. 定义 Project、SourceVideo、OperationRun 数据对象；为项目、视频和运行生成稳定唯一 ID，并定义 queued/running/completed/cancelled/failed 状态。
2. 实现项目根目录创建和项目名安全化；新导入批次创建独立目录，绝不复用已有项目目录。
3. 实现 `project.json` 的版本化读写，保存源视频绝对路径、视频基本信息、FPS 设置、每次操作状态、计数和输出路径。
4. 实现重新打开时检测源视频路径；路径失效则在数据模型中标记 unavailable，并提供更新视频路径的方法。
5. 编写临时目录测试：项目隔离、JSON 往返、重名新建不覆盖、无效视频路径状态和重新关联。

**验收：** 项目元数据可往返读取，任何已有项目目录都不会被覆盖。

### 阶段 2：实现可复用数据服务与 CLI 兼容层

**文件：** `dataset_tool/services/video_extractor.py`、`dataset_tool/services/labeled_splitter.py`、`dataset_tool/services/image_sampler.py`、三个现有脚本、对应服务测试

1. 为三项服务定义结构化结果和统一的输入检查；输入不存在、参数非法、输出目录非空或文件写入失败时返回可供 UI/CLI 展示的明确错误。
2. 抽帧服务接受视频路径、目标 FPS、输出目录、进度回调和取消回调；按 `max(1, int(source_fps / target_fps))` 采样，从第 0 帧开始，保存为按序编号的 JPG，并确保任何退出路径都释放 OpenCV 资源。
3. 抽帧服务对无法打开、无有效 FPS、无帧或写入失败的视频报告失败；目标 FPS 大于源 FPS 时逐帧保存。
4. 带标签划分仅收集小写 `.jpg`；仅将同名 `.txt` 存在的配对纳入随机划分；空标签文件有效，缺失标签计入警告并跳过。默认 70/20/10、seed 42；比例非负且总和大于零，非 1 时归一化并返回提示。
5. 图片抽样服务默认抽样比例 20%、train 比例 70%、seed 42，只复制图片到 images/train 和 images/val；返回“未复制标签”的说明。
6. 所有服务创建目录前检查目标为空；取消时中止后续处理并保留已完成输出。服务不得移动或修改源文件。
7. 改造三个脚本为 CLI 适配层，保持现有参数/缺省值兼容，并将服务错误显示给命令行用户。
8. 使用临时图片、标签和短视频验证取帧规则、命名、配对、比例/seed 可重复性、取消、坏输入、非空目录拒绝及源文件保留。

**验收：** GUI 与三个 CLI 入口调用相同服务；服务测试只在临时目录内写入。

### 阶段 3：实现后台任务与模块注册

**文件：** `dataset_tool/workers/operation_worker.py`、`dataset_tool/modules/base.py`、`dataset_tool/modules/registry.py`

1. 定义可用模块与未接入模块的最小接口，注册视频抽帧、带标签划分、图片抽样、标注占位和训练占位。
2. 实现 QThread worker：接收一个服务调用及参数，发出 progress、completed、cancelled、failed 信号。
3. 实现线程安全取消标记；服务回调检查该标记，结束后始终整理线程和控件状态。
4. 为 worker 信号映射及取消状态转换编写轻量测试或可重复的 GUI 验收步骤。

**验收：** 耗时任务不在 UI 线程执行；取消和失败都不会显示为成功。

### 阶段 4：实现项目与视频抽帧页面

**文件：** `dataset_tool/ui/main_window.py`、`dataset_tool/ui/pages/video_page.py`、`dataset_tool/ui/widgets/thumbnail_gallery.py`

1. 搭建中文主窗口和导航；左侧显示项目/视频列表，中间显示视频预览，右侧显示目标 FPS、输出信息、进度和取消按钮，底部展示缩略图。
2. 支持新建项目、为一次导入选择多个视频、移除列表项、重新关联丢失的视频路径；只记录源视频路径，不复制视频。
3. 用 OpenCV 获取视频首帧预览和基本信息；错误或失效路径以中文状态提示呈现。
4. 每次抽帧新建 extraction run 和按视频隔离的 frames 子目录；显示各视频进度与抽帧结果，禁止复用非空输出目录。
5. 支持点击缩略图查看大图，并使用系统文件管理器打开对应图片或目录。
6. 用户重新打开项目时恢复视频列表、抽帧记录和已有缩略图；再次抽帧创建新的 run。
7. 手动启动应用，验证新建项目、多视频导入、预览、抽帧、缩略图、取消、失败提示和重复运行。

**验收：** 界面保持响应；项目数据可重开；抽帧图片始终写入项目的独立运行目录。

### 阶段 5：实现两种划分页及占位入口

**文件：** `dataset_tool/ui/pages/labeled_split_page.py`、`dataset_tool/ui/pages/image_sample_page.py`、`dataset_tool/ui/pages/unavailable_page.py`、`dataset_tool/ui/main_window.py`

1. 带标签页面提供图片目录、标签目录、输出位置、train/val/test 比例及随机种子输入，显示配对数、缺失标签数、实际比例和运行结果。
2. 图片抽样页面提供图片目录、输出位置、抽样比例、train/val 比例及随机种子输入，明确说明标签不会复制。
3. 两页都可选择把输出放进当前项目的新 datasets run 目录，或选定其他空目录；非空目录拒绝运行。
4. 两页均使用阶段 3 worker 调用服务，提供进度、取消和中文错误提示；取消时保留已写结果并标明“已取消”。
5. 标注和训练导航打开简洁占位页，显示“未接入”，不启动外部工具。
6. 手动验证两页参数、结果统计、非空输出保护、取消行为和占位导航。

**验收：** 两种划分保持独立、服务不修改源目录；image-sample 只包含图片。

### 阶段 6：启动说明与完整验收

**文件：** `README.md`、必要的界面/服务修正

1. 编写中文启动说明：运行环境、依赖、`python app.py`、项目目录、各页面行为、取消后文件状态和 CLI 用法。
2. 在隔离的临时目录运行全部自动化测试；记录使用的命令和结果。
3. 手动启动 PyQt5 应用检查页面导航、预览、任务状态、进度和取消；验收样本只放在临时目录。
4. 对现有 `images/`、`labels/`、`iceice/`、`xz_dataset/` 做只读数量与代表性文件哈希比对，确认开发过程未改动它们。
5. 修正验收中发现的问题，复跑相关临时测试并确认 GUI 可启动。

**最终验收：** 中文说明完整；所有目标功能工作；现有数据集保持原样；三个旧脚本的 CLI 入口仍可使用。

---

## 执行顺序

按阶段 1 到阶段 6 顺序实施。每阶段完成后先检查本阶段文件和验收结果，再进入下一阶段；若实现细节与已确认设计冲突，以本计划和中文设计文档为准。

