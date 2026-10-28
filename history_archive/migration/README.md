# 2026-10-28 归档搬迁记录

- [manifest.json](manifest.json)：7729个文件、3,849,878,873字节的原路径、英文目标路径和SHA-256。
- [tracked_before.json](tracked_before.json)：搬迁前源码跟踪清单，用于保持Git存档边界。
- [protected_files.json](protected_files.json)：自动关机脚本、伪深度模型锁文件与两个运行权重的搬迁前哈希。
- prediction_shortcut_before.lnk：桌面预测结果快捷方式原件备份。
- original_documents/：更新历史索引链接前的Markdown/JSON原件；文件名使用原件SHA-256。
- verification.json：验收结果（完成后写入）。

每个文件搬迁时已核对大小与SHA-256。少量Markdown/索引JSON随后更新路径，manifest中另列current_sha256与原件备份；原始训练args.yaml、实验参数、CSV、权重和日志不改写。

两个原Windows Junction已展开为独立目录；清单materialized_junctions及materialized_from记录其来源。所有新目标路径均为ASCII。这里的审计JSON、快捷方式和原件备份按现有存档规则只在本地保留。
