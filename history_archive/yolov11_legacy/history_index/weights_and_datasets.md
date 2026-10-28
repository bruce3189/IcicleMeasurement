> 历史实验归档，不是当前采用模型。当前版本见 [最终成果](<../../../final_results/README.md>)。

# 权重与数据集索引

## 历史权重与当时用途（非当前默认模型）

- 检测入口 mypredict.py 当前引用 [runs/detect/train10/weights/best.pt](<../runs/detect/train10/weights/best.pt>)；train10 的 CSV 末轮 Box mAP50-95 为 0.70092，历史峰值为第 91 轮 0.70616。这个峰值不等同于已重新验证 best.pt 的成绩。
- 历史三点 Pose 基线：[runs_pose/ice80_yolo26/weights/best.pt](<../runs_pose/ice80_yolo26/weights/best.pt>)。
- 历史三点 Pose tip 加权：[runs_pose/ice80_yolo26_tipw/weights/best.pt](<../runs_pose/ice80_yolo26_tipw/weights/best.pt>)。
- 两点 Pose 属于 Measurement 的另一条训练线，现有归档在 [final_results/training/run](<../../../final_results/training/run>)；本次没有迁移或替换该模型。

根目录 yolo11n.pt/yolo11s.pt 与 weights/yolo11n-pose.pt、weights/yolo26n-pose.pt 是初始权重文件；历史实验中的 best.pt/last.pt 单列如下。同名、同大小不能证明内容相同，SHA-256 可用来核对文件。

## 权重文件清单

| 路径 | 大小 MiB | SHA-256 |
|---|---:|---|
| [runs/detect/train/weights/best.pt](<../runs/detect/train/weights/best.pt>) | 54.63 | `811090dc8a357a46d6c39e720643846201eabf3d6d6b90e55eb2fc8616c9142e` |
| [runs/detect/train/weights/last.pt](<../runs/detect/train/weights/last.pt>) | 54.63 | `811090dc8a357a46d6c39e720643846201eabf3d6d6b90e55eb2fc8616c9142e` |
| [runs/detect/train10/weights/best.pt](<../runs/detect/train10/weights/best.pt>) | 5.20 | `f1151673613d1caf5241937635e443c954de702e7b4a5ffa0266b87554d2407e` |
| [runs/detect/train10/weights/last.pt](<../runs/detect/train10/weights/last.pt>) | 5.20 | `3085eee0fb864f71514eeea39b454128382a1174f26bdb68bc963230cf980649` |
| [runs/detect/train2/weights/best.pt](<../runs/detect/train2/weights/best.pt>) | 15.32 | `47093ec9f233b2713a04144dafceee7b2afa3d17476128509f930eb548c2f1c0` |
| [runs/detect/train2/weights/last.pt](<../runs/detect/train2/weights/last.pt>) | 15.32 | `47093ec9f233b2713a04144dafceee7b2afa3d17476128509f930eb548c2f1c0` |
| [runs/detect/train3/weights/best.pt](<../runs/detect/train3/weights/best.pt>) | 5.21 | `fc047a4e475d1111eb2bb2cde2d1f690a78bd57f67b758cfe2947a87d8f5ea86` |
| [runs/detect/train3/weights/last.pt](<../runs/detect/train3/weights/last.pt>) | 5.21 | `b2251e60f75d4999eeadf9ee65be6f8992402e8578504c67d4cff109f6b7d149` |
| [runs/detect/train4/weights/best.pt](<../runs/detect/train4/weights/best.pt>) | 5.21 | `b105989c52e3be278d6fa8e06713f55d7b35b44c1020407c34f158ba128d872e` |
| [runs/detect/train4/weights/last.pt](<../runs/detect/train4/weights/last.pt>) | 5.21 | `109c89a22f5f7cf7adffa52f04d6cdd1c25468953b1fc5af38163f48e93077ec` |
| [runs/detect/train6/weights/best.pt](<../runs/detect/train6/weights/best.pt>) | 5.21 | `11c15500f80b7de47adc6b06b9465e0aead8439b09944be6f297f2811bd3771b` |
| [runs/detect/train6/weights/last.pt](<../runs/detect/train6/weights/last.pt>) | 5.21 | `975d51726f8e89282b122008cddf2216107b05767e656126d8390e282a26e2b7` |
| [runs/detect/train7/weights/best.pt](<../runs/detect/train7/weights/best.pt>) | 5.22 | `9d9e69540d6aa1541c28ccff8f619478be93b324e74785882a880988ece42f91` |
| [runs/detect/train7/weights/last.pt](<../runs/detect/train7/weights/last.pt>) | 5.22 | `055d8c8d216d29290228649172b67216589786d7792078d2700f700362e5b4d0` |
| [runs/detect/train8/weights/best.pt](<../runs/detect/train8/weights/best.pt>) | 5.23 | `51f1c48349e6c51bfcdd88989b298bab3f552bf415c9517a8fcbe25e1a2c2f16` |
| [runs/detect/train8/weights/last.pt](<../runs/detect/train8/weights/last.pt>) | 5.23 | `286f6921e9e056341eac147228b4819a7ab393c776c8e9ad5104e58bb826b1dd` |
| [runs/detect/train9/weights/best.pt](<../runs/detect/train9/weights/best.pt>) | 5.19 | `db128cee8442d64a51932939d7612633d73389a1d716f4c2851992f631f39ed9` |
| [runs/detect/train9/weights/last.pt](<../runs/detect/train9/weights/last.pt>) | 5.19 | `6689ef3baa8f32ccd096f6f47bea3c30b8038500622f48247d34f62019132f77` |
| [runs_pose/control_tip1_232_b1_10ep/weights/best.pt](<../runs_pose/control_tip1_232_b1_10ep/weights/best.pt>) | 5.62 | `363d7b8fb64095178031816a2d9aad52911f5fabca11f9e0c21be1d9dc06aae4` |
| [runs_pose/control_tip1_232_b1_10ep/weights/last.pt](<../runs_pose/control_tip1_232_b1_10ep/weights/last.pt>) | 5.62 | `abc2def5a135471376042eb54e0561f42b99586f2d65952f9cd63595ca0b4fd0` |
| [runs_pose/ice80_yolo26/weights/best.pt](<../runs_pose/ice80_yolo26/weights/best.pt>) | 5.64 | `b326ca57763966652ed3f77efe184d6126110543af0c728b5f1a34b3e5105307` |
| [runs_pose/ice80_yolo26/weights/last.pt](<../runs_pose/ice80_yolo26/weights/last.pt>) | 5.64 | `0dc6fa5349aea872f4108206057f67a1f09693162dc95b55eac94ac81436cc97` |
| [runs_pose/ice80_yolo26_tipw/weights/best.pt](<../runs_pose/ice80_yolo26_tipw/weights/best.pt>) | 5.64 | `9e37286db87b69333db1a1e18fd7fa8e9b52229f45ee3380e4544427f0a22df3` |
| [runs_pose/ice80_yolo26_tipw/weights/last.pt](<../runs_pose/ice80_yolo26_tipw/weights/last.pt>) | 5.64 | `72010e868af6bc1c84b163cd229b97f18b83aec277cba5875555e8cda3726af3` |
| [runs_pose/smoke_yolo26/weights/best.pt](<../runs_pose/smoke_yolo26/weights/best.pt>) | 5.53 | `bc76ac9895f551bc93e4709fa9f6946a9f88619694e52c36d5b5b880094c984d` |
| [runs_pose/smoke_yolo26/weights/last.pt](<../runs_pose/smoke_yolo26/weights/last.pt>) | 5.53 | `449daaa0396fa37d7b9cb95fb117a59038fbabb9326da60aee01b5006bddbcc5` |
| [runs_pose/tip3_232_b1_10ep/weights/best.pt](<../runs_pose/tip3_232_b1_10ep/weights/best.pt>) | 5.62 | `c4ebadbfc4cd3a086dc2f3deadcbcb730e8740f4450df8918f43e008ebda5701` |
| [runs_pose/tip3_232_b1_10ep/weights/last.pt](<../runs_pose/tip3_232_b1_10ep/weights/last.pt>) | 5.62 | `0533c5de1d88d91ef55296a1bd69f4f297c175fcb194044011750ccf75043837` |
| [runs_pose/tip_refine/experiments/control_tip1_b1_10ep/weights/best.pt](<../runs_pose/tip_refine/experiments/control_tip1_b1_10ep/weights/best.pt>) | 5.62 | `cd01178862ad903f76064c41867fd72e0f0ac98f1342497e8a0c9d4739e1a6ec` |
| [runs_pose/tip_refine/experiments/control_tip1_b1_10ep/weights/last.pt](<../runs_pose/tip_refine/experiments/control_tip1_b1_10ep/weights/last.pt>) | 5.62 | `554a2ecd945558f279a53ed05aa1bfff57f2da6d0835434525c6878ba7a835bf` |
| [runs_pose/tip_refine/experiments/head_control_tip1_10ep/weights/best.pt](<../runs_pose/tip_refine/experiments/head_control_tip1_10ep/weights/best.pt>) | 5.62 | `5a0ff13dd24fdc0ac43eb6439e8279e2c4ec277adcb260a185d76032a057aef8` |
| [runs_pose/tip_refine/experiments/head_control_tip1_10ep/weights/last.pt](<../runs_pose/tip_refine/experiments/head_control_tip1_10ep/weights/last.pt>) | 5.62 | `14b97317ff34149546f88203bd6eb021a6ae364eb28757c104fbeafc11b9630a` |
| [runs_pose/tip_refine/experiments/head_direct_tip_10ep/weights/best.pt](<../runs_pose/tip_refine/experiments/head_direct_tip_10ep/weights/best.pt>) | 7.27 | `035fddc5517563d21790a59b17c329dcb82edaf73bc3b985b7e114eb9e67eef8` |
| [runs_pose/tip_refine/experiments/head_direct_tip_10ep/weights/last.pt](<../runs_pose/tip_refine/experiments/head_direct_tip_10ep/weights/last.pt>) | 7.27 | `6a811bc7ce57cad15acee726ee9ea5d32b009aea7834e07126735ea70d7156c3` |
| [runs_pose/tip_refine/experiments/head_tip3_10ep/weights/best.pt](<../runs_pose/tip_refine/experiments/head_tip3_10ep/weights/best.pt>) | 5.62 | `7afd109d9a363f97e0c7fb192c27d560deeed242472211b3534cc7a7729c9fc5` |
| [runs_pose/tip_refine/experiments/head_tip3_10ep/weights/last.pt](<../runs_pose/tip_refine/experiments/head_tip3_10ep/weights/last.pt>) | 5.62 | `3d56c44eb5faac5b9642d9199c462945bd99e70537d87fe0f4a29c328be8d172` |
| [runs_pose/tip_refine/experiments/smoke_tip3_v2/weights/best.pt](<../runs_pose/tip_refine/experiments/smoke_tip3_v2/weights/best.pt>) | 5.62 | `1213cac2f3b7633cf405cb5bab8023e3db03a4a32e846494596e56e3879a067d` |
| [runs_pose/tip_refine/experiments/smoke_tip3_v2/weights/last.pt](<../runs_pose/tip_refine/experiments/smoke_tip3_v2/weights/last.pt>) | 5.62 | `aef2578e656ff2be1c3708a5db4feaad935622aac9e4e1722807cef688090a17` |
| [runs_pose/tip_refine/experiments/tip3_b1_10ep/weights/best.pt](<../runs_pose/tip_refine/experiments/tip3_b1_10ep/weights/best.pt>) | 5.62 | `5c858a28dee9ae703a5709619c4c422580275e29167d7ec3ea9ba43260a09e7e` |
| [runs_pose/tip_refine/experiments/tip3_b1_10ep/weights/last.pt](<../runs_pose/tip_refine/experiments/tip3_b1_10ep/weights/last.pt>) | 5.62 | `65c9ccd69bf0de8764a7344d37173244120ff345374e3dc20b89f3316edfb469` |
| [runs_pose/tip_refine_232/experiments/control_tip1_232_b1_10ep/weights/best.pt](<../runs_pose/tip_refine_232/experiments/control_tip1_232_b1_10ep/weights/best.pt>) | 5.62 | `363d7b8fb64095178031816a2d9aad52911f5fabca11f9e0c21be1d9dc06aae4` |
| [runs_pose/tip_refine_232/experiments/control_tip1_232_b1_10ep/weights/last.pt](<../runs_pose/tip_refine_232/experiments/control_tip1_232_b1_10ep/weights/last.pt>) | 5.62 | `abc2def5a135471376042eb54e0561f42b99586f2d65952f9cd63595ca0b4fd0` |
| [runs_pose/tip_refine_232/experiments/tip3_232_b1_10ep/weights/best.pt](<../runs_pose/tip_refine_232/experiments/tip3_232_b1_10ep/weights/best.pt>) | 5.62 | `c4ebadbfc4cd3a086dc2f3deadcbcb730e8740f4450df8918f43e008ebda5701` |
| [runs_pose/tip_refine_232/experiments/tip3_232_b1_10ep/weights/last.pt](<../runs_pose/tip_refine_232/experiments/tip3_232_b1_10ep/weights/last.pt>) | 5.62 | `0533c5de1d88d91ef55296a1bd69f4f297c175fcb194044011750ccf75043837` |
| [weights/yolo11n-pose.pt](<../weights/yolo11n-pose.pt>) | 5.97 | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` |
| [weights/yolo26n-pose.pt](<../weights/yolo26n-pose.pt>) | 7.51 | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |
| [yolo11n.pt](<../yolo11n.pt>) | 5.35 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| [yolo11s.pt](<../yolo11s.pt>) | 18.42 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |

## 当前数据快照

| 目录 | train 图片/标签 | val 图片/标签 | test 图片/标签 |
|---|---:|---:|---:|
| [datasets/iceice](<../../../Yolov11/datasets/iceice>) | 33/33 | 9/9 | 6/6 |
| [datasets/xz_dataset](<../../../Yolov11/datasets/xz_dataset>) | 28/29 | 15/16 | 0/0 |
| [runs_pose/tip_refine/data_snapshot](<../runs_pose/tip_refine/data_snapshot>) | 50/50 | 20/20 | 10/10 |
| [runs_pose/tip_refine_232/data_snapshot](<../runs_pose/tip_refine_232/data_snapshot>) | 169/169 | 38/38 | 25/25 |

图片/标签计数为当前快照，不证明历史训练时的数据内容完全相同。xz_dataset 的标签文件数包含 train/val 各一份 classes.txt，实际逐图标签分别为 28/15。Easy80 为 50/20/10，232 张快照为 169/38/25。后者原说明记录分组为连续 4 帧整块分配；这仍不等同于按整个视频或拍摄场景隔离。

检测数据配置现已确认存在：[ultralytics/cfg/datasets/ice_xz.yaml](<../../../Yolov11/ultralytics/cfg/datasets/ice_xz.yaml>) 和 [ultralytics/cfg/datasets/iceice.yaml](<../../../Yolov11/ultralytics/cfg/datasets/iceice.yaml>)。根目录没有 iceice.yaml 不代表本地源码内也没有；mytrain.py 传入相对名。配置中的数据根路径为 xz_dataset/iceice，是否按预期解析取决于运行时 Ultralytics 设置，本次未启动训练验证。

## 路径与复现说明

原始 args.yaml、experiment.json、source_manifest.json 以及历史训练脚本保留了 D:\yolov11、桌面目录等来源路径，未改写成当前副本路径。直接重新运行旧 Pose 脚本可能写入旧目录；本索引是历史查询入口，不是已经适配好的重训入口。历史运行说明中的“已启动”不代表当前仍在训练。

训练目录和权重已集中到 history_archive/yolov11_legacy；重复记录完整保留。未重新训练或推理。Git 当前规则仍排除权重、图片、日志及 JSON 清单；Markdown 索引可进入源码版本管理，二进制归档需要保留本地目录。
