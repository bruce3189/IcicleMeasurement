"""图片抽样划分命令行入口；标签不参与处理，源图片只复制不移动。"""

import argparse
from pathlib import Path

from dataset_tool.services.image_sampler import sample_images

ROOT_DIR = Path(__file__).resolve().parent


def split_and_copy_images(source_folder=None, output_dir=None, sample_ratio=0.2,
                          train_ratio=0.7, random_seed=42):
    """兼容原函数入口，抽样图片并按 train/val 比例复制。"""
    result = sample_images(source_folder or ROOT_DIR / "images", output_dir or ROOT_DIR / "xz_dataset",
                           sample_ratio, train_ratio, random_seed,
                           progress_callback=lambda event: print("复制进度：{}/{} ({})".format(event["current"], event["total"], event["split"])))
    print("完成：{}，各集合数量：{}。标签未复制。".format(result["output_dir"], result["counts"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="随机抽样图片并划分 train/val（不处理标签）")
    parser.add_argument("--images", type=Path, default=ROOT_DIR / "images", help="源图片目录")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "xz_dataset", help="空的输出目录")
    parser.add_argument("--sample-ratio", type=float, default=0.2, help="抽样比例，默认 0.2")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="抽样图片中训练集比例，默认 0.7")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args(argv)
    try:
        split_and_copy_images(args.images, args.output, args.sample_ratio, args.train_ratio, args.seed)
        return 0
    except Exception as error:
        parser.exit(1, "错误：{}\n".format(error))


if __name__ == "__main__":
    raise SystemExit(main())
