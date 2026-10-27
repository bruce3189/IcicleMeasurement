"""图片抽样并划分 train/val 的服务；不处理标签。"""

import random
import shutil
import math
from pathlib import Path

from .common import check_cancel, emit_progress, prepare_empty_output


def sample_images(source_dir, output_dir, sample_ratio=0.2, train_ratio=0.7, random_seed=42,
                  progress_callback=None, cancel_callback=None):
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError("图片目录不存在：{}".format(source))
    sample_ratio, train_ratio = float(sample_ratio), float(train_ratio)
    if not math.isfinite(sample_ratio) or not 0 < sample_ratio <= 1:
        raise ValueError("抽样比例必须大于 0 且不超过 1")
    if not math.isfinite(train_ratio) or not 0 <= train_ratio <= 1:
        raise ValueError("训练集比例必须在 0 到 1 之间")
    # make_xz.py 原先只纳入小写 .jpg；GUI 保持这一筛选约定。
    candidates = sorted(path for path in source.iterdir() if path.is_file() and path.suffix == ".jpg")
    if not candidates:
        raise ValueError("图片目录中没有支持的图片文件")
    count = int(len(candidates) * sample_ratio)
    if count < 1:
        raise ValueError("当前图片数量太少，按该抽样比例无法选出图片")
    selected = random.Random(random_seed).sample(candidates, count)
    train_count = int(count * train_ratio)
    partitions = {"train": selected[:train_count], "val": selected[train_count:]}
    output = prepare_empty_output(output_dir)
    copied = 0
    for split_name, subset in partitions.items():
        destination = output / "images" / split_name
        destination.mkdir(parents=True, exist_ok=True)
        for image in subset:
            check_cancel(cancel_callback)
            shutil.copy2(str(image), str(destination / image.name))
            copied += 1
            emit_progress(progress_callback, current=copied, total=count, split=split_name, filename=image.name)
    return {"selected_count": count, "counts": {name: len(items) for name, items in partitions.items()},
            "sample_ratio": sample_ratio, "train_ratio": train_ratio, "random_seed": random_seed,
            "output_dir": str(output), "labels_copied": False}
