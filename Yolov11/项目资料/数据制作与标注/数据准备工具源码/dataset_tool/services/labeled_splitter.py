"""带标签的 JPG/TXT 配对划分服务。"""

import random
import shutil
import math
from pathlib import Path

from .common import check_cancel, emit_progress, prepare_empty_output


def _normalized_ratios(train_ratio, val_ratio, test_ratio):
    values = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("划分比例必须非负，且总和必须大于 0")
    total = sum(values)
    warning = None
    if abs(total - 1.0) > 1e-9:
        warning = "比例总和为 {:.6g}，已自动归一化。".format(total)
        values = [value / total for value in values]
    return values, warning


def split_labeled_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2,
                         test_ratio=0.1, random_seed=42, progress_callback=None, cancel_callback=None):
    images = Path(images_dir)
    labels = Path(labels_dir)
    if not images.is_dir():
        raise FileNotFoundError("图片目录不存在：{}".format(images))
    if not labels.is_dir():
        raise FileNotFoundError("标签目录不存在：{}".format(labels))
    ratios, warning = _normalized_ratios(train_ratio, val_ratio, test_ratio)
    image_paths = sorted(path for path in images.iterdir() if path.is_file() and path.suffix == ".jpg")
    if not image_paths:
        raise ValueError("图片目录中没有小写 .jpg 图片")
    pairs, missing = [], []
    for path in image_paths:
        label = labels / (path.stem + ".txt")
        if label.is_file():
            pairs.append((path, label))
        else:
            missing.append(path.name)
    if not pairs:
        raise ValueError("没有找到图片和同名 TXT 标签的有效配对")
    random.Random(random_seed).shuffle(pairs)
    train_count = int(len(pairs) * ratios[0])
    val_count = int(len(pairs) * ratios[1])
    partitions = [pairs[:train_count], pairs[train_count:train_count + val_count], pairs[train_count + val_count:]]
    output = prepare_empty_output(output_dir)
    total = len(pairs)
    copied = 0
    for split_name, subset in zip(("train", "val", "test"), partitions):
        image_dest = output / "images" / split_name
        label_dest = output / "labels" / split_name
        image_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)
        for image, label in subset:
            check_cancel(cancel_callback)
            shutil.copy2(str(image), str(image_dest / image.name))
            shutil.copy2(str(label), str(label_dest / label.name))
            copied += 1
            emit_progress(progress_callback, current=copied, total=total, split=split_name, filename=image.name)
    info = output / "split_info.txt"
    with info.open("w", encoding="utf-8") as stream:
        stream.write("数据集划分信息\n总配对数：{}\n随机种子：{}\n".format(total, random_seed))
        stream.write("比例：train {:.6f}, val {:.6f}, test {:.6f}\n".format(*ratios))
        stream.write("缺少标签而跳过：{}\n".format(len(missing)))
        for name, subset in zip(("train", "val", "test"), partitions):
            stream.write("\n{} ({}):\n".format(name, len(subset)))
            for image, _ in subset:
                stream.write(image.name + "\n")
    return {"total_pairs": total, "counts": {name: len(part) for name, part in zip(("train", "val", "test"), partitions)},
            "missing_labels": missing, "ratios": ratios, "warning": warning, "output_dir": str(output)}
