"""把 232 张合格数据集快照到实验工作区（只读源，生成独立 dataset.yaml + SHA256 清单）。"""
import hashlib, json, shutil
from pathlib import Path

SRC = Path(r"C:\Users\Administrator\Desktop\Iceicle_数据集_400_20261023")
HOME = Path(r"D:\yolov11\runs_pose\tip_refine_232")
DST = HOME / "data_snapshot"
SPLITS = ("train", "val", "test")

if DST.exists():
    raise SystemExit(f"快照已存在，不覆盖：{DST}（要换数据请先手动改名留档）")

manifest = {"source": str(SRC), "splits": {}, "files": {}}
for sp in SPLITS:
    (DST / "images" / sp).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / sp).mkdir(parents=True, exist_ok=True)
    imgs = sorted((SRC / "images" / sp).glob("*.jpg"))
    for im in imgs:
        lb = SRC / "labels" / sp / f"{im.stem}.txt"
        if not lb.exists():
            raise SystemExit(f"缺标签：{im}")
        shutil.copy2(im, DST / "images" / sp / im.name)
        shutil.copy2(lb, DST / "labels" / sp / lb.name)
        manifest["files"][f"{sp}/{im.name}"] = hashlib.sha256(im.read_bytes()).hexdigest()[:16]
        manifest["files"][f"{sp}/{lb.name}"] = hashlib.sha256(lb.read_bytes()).hexdigest()[:16]
    manifest["splits"][sp] = len(imgs)

(DST / "dataset.yaml").write_text(
    f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: ice\nkpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\n"
    "kpt_names:\n  0: [tip, base_left, base_right]\n", encoding="utf-8")
(DST / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(manifest["splits"], ensure_ascii=False),
      "| 文件数", len(manifest["files"]), "| 快照:", DST)
