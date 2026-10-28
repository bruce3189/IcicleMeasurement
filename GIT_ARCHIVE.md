# 本地 Git 存档范围

这是 D:\IcicleMeasurement 的代码版本存档，不是完整运行环境备份。
本次只新增 .gitignore、本说明和隐藏的 .git 历史目录，没有改写、移动或删除原项目文件。

## 首次存档

- 分支 main；署名 Bruce <3391263639@qq.com>，仅配置在本仓库。
- 提交说明：存档：冰凌测距项目初始版本。
- 原有文件纳入 1,104 个（8,037,300 字节），另纳入 .gitignore 与本说明，共 1,106 个文件。
- 排除 16,803 个原有文件（8,116,625,302 字节）；全部仍保存在原位置。
- 没有添加 GitHub remote，没有推送；提交邮箱会随将来的提交上传。
- 本仓库关闭自动 CRLF 转换，暂存内容逐文件与工作区原始字节核对。

## 纳入与排除

纳入 Measurement 与 Yolov11 的代码、测试、依赖说明、配置模板、启动入口和现存第三方许可证。
work/pseudo_models 中的固定版本模型源码与 models.lock.json 保留；虚拟环境、下载压缩包和权重排除。
outputs/ice_pose_runs 中的预测、训练、看板等源代码与必要 dataset.yaml 保留；运行输出、标签和数据排除。
runs_pose 中的源脚本和数据集配置保留；历史实验输出、数据、权重排除。

排除：原始照片、视频、训练标注与数据、深度 NPY/PNG、模型权重、虚拟环境、缓存、日志、历史备份、
导出数据副本、IDE 状态、迁移审计备份及未明确纳入的二进制文件。
忽略规则采用源文件类型白名单；以后新增 JSON 等未纳入类型时，应先检查用途，再精准加入规则。
忽略并不删除文件；它们也不受 Git 版本恢复保护。

## 运行资源与外部依赖

- 本机基础 Python/依赖仍位于 D:\318；Git 位于 D:\Git，未复制进仓库。
- Measurement/assets/rgb 的400张原图、outputs 下的预测标签，以及 training/pose_2kpt/runs 下的两点 Pose 权重和运行记录仅在本机保留。
- work/pseudo_env、模型权重需另行准备；setup_pseudo_depth.py 保留原有固定版本设置，本次没有运行安装。
- 两点 Pose 的训练运行归档在 Measurement/training/pose_2kpt/runs/yolo11n_pose_78_2kpt；训练数据集仍位于 outputs/ice_pose_runs/dataset_冰凌_两点_78_20261023。训练脚本使用项目相对路径，新运行使用时间戳目录以避免覆盖归档。
- 部分历史脚本仍有外部数据路径；mytrain.py 所需 iceice.yaml 原本缺失，没有替换数据集或启动训练。
- 第三方许可证按现有文件保留，没有给第三方代码重新授权；将来公开发布前仍需核对分发要求。
- 当前代码、注释及文档可能包含本地路径；规则凭据扫描没有发现待处理的凭据，但不代替公开发布审查。
- 合成深度与模拟结果不代表真实冰凌测量精度。

## 本地清单与验收记录

完整纳入/排除清单、原文件SHA256和验收结果保存在 .git/archive-audit/，不进入提交。
该目录是本机审计附件；克隆仓库不会自动包含它。另有一份保存在
C:\Users\Administrator\Documents\Codex\git_archive_preparation_20261027。
本次不运行应用、测试、训练或自动格式化，以免产生运行副作用。

## 以后存档

在 Git Bash 中进入 /d/IcicleMeasurement：

```bash
git status
git add .
git diff --cached --stat
git commit -m "描述本次修改"
git log --oneline -5
```

先核对暂存清单，再提交。以上命令不会自动上传，也不会保存被忽略的权重、图片和环境。
恢复历史版本前先检查未提交修改，避免覆盖正在进行的工作。

## 2026-10-28 本机训练记录整理

两点 Pose 训练运行（30个文件，30,969,017字节）已从 `Measurement/outputs/ice_pose_runs/yolo11n_pose_78_2kpt` 移到 `Measurement/training/pose_2kpt/runs/yolo11n_pose_78_2kpt`；训练脚本移到同目录的上层。训练权重和运行结果仍由 `.gitignore` 排除，训练脚本作为源码保留在 Git 工作区。
默认 Pose 权重、预测入口、看板、训练脚本和使用说明已经改用新路径。训练数据集仍在 `Measurement/outputs/ice_pose_runs/dataset_冰凌_两点_78_20261023`。历史 `args.yaml` 内绝对路径保留作训练来源记录。
路径与训练脚本的代码变更现已包含在提交 `1b23813`，本地分支与 `origin/main` 一致；运行记录和权重仍未纳入 Git。该提交由本机 Git 记录的署名邮箱为 `sabanarupeali@gmail.com`。
`Measurement/outputs/ice_pose_runs/shutdown_after_training.ps1` 按要求原样留在原位置且未运行；脚本包含 Windows 自动关机命令，请勿启动。
