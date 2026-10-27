# 数据集准备工具

这是一个中文 PyQt5 桌面工具，用于创建独立视频项目、抽取视频帧，以及运行两种互相独立的数据集划分流程。标注和训练入口当前显示为“未接入”，只保留页面和模块接口。

## 环境与启动

使用 Python 3.9 或更新版本。在项目目录打开终端运行：

```powershell
python -m pip install -r requirements-gui.txt
python app.py
```

GUI 也使用 OpenCV。若在隔离环境中运行，请确认 `opencv-python` 与 PyQt5 安装在启动 `app.py` 的同一个 Python 环境中。

## 构建 Windows exe

在 Windows 上使用当前项目的 Python 环境运行：

```powershell
python -m pip install -r requirements-build.txt
.\build_exe.ps1
```

生成的单文件程序位于 `dist\DatasetTool.exe`。构建不打包 `images/`、`labels/`、`iceice/`、`xz_dataset/`、`videos/` 或项目文件；exe 首次运行时会在其所在目录创建 `projects/`。发布时可以只分发 exe，用户的视频和数据仍由界面选择。

## 视频项目和抽帧

在“视频抽帧”页新建项目并选择一个或多个视频。原始视频不会复制，项目只记录其绝对路径。默认每秒抽取 2 帧，应用从视频第 0 帧开始按 `max(1, int(源 FPS / 目标 FPS))` 的间隔采样。每次抽帧在项目 `extraction_runs/` 中创建独立目录，已有结果不会被覆盖。取消时已写出的帧会保留，运行记录标记为已取消。视频移动后，可在重新打开项目时重新关联路径。

抽出的图片显示在页面下方缩略图区域。单击可以预览，双击可打开图片所在目录，之后可用你选择的标注工具手工标注。

## 数据划分页面

- **图片和标签划分**：选择图片目录、标签目录和输出目录。默认按 70/20/10 划分，固定随机种子 42。只复制小写 `.jpg` 和同名 `.txt`；缺少标签的图片会跳过并报告，空标签仍是有效标签。
- **图片抽样划分**：保持原脚本规则，只从小写 `.jpg` 图片中抽样 20%，然后按 70/30 划分 train/val。此页面仅复制图片，不会复制标签。
- 两种流程都只复制源文件。输出目录必须为空；非空目录会被拒绝。

输出目录留空时，如果当前已打开项目，结果保存在项目的 `datasets/` 下新建运行目录；没有打开项目时，会创建在 `projects/standalone_runs/` 下。

## 旧命令行入口

三个旧脚本保留 CLI 入口，并与 GUI 使用相同服务：

```powershell
python extract.py --fps 2 --video_dir .\videos --output_dir .\new_frames
python split.py --images .\images --labels .\labels --output .\new_labeled_split
python make_xz.py --images .\images --output .\new_image_sample
```

输出目录必须为空或尚不存在。默认路径相对脚本位置；若默认目录已有数据，脚本会停止以保护已有内容。`make_xz.py` 不会读取标签。

## 项目目录

```text
projects/
  <项目 ID>-<项目名称>/
    project.json
    extraction_runs/<运行类型>-<运行 ID>/frames/<视频 ID>-<视频名>/*.jpg
    datasets/<运行类型>-<运行 ID>/...
```

`images/`、`labels/`、`iceice/` 和 `xz_dataset/` 仍是独立的现有数据目录。应用不会自动导入、合并、重新划分或覆盖这些数据。
