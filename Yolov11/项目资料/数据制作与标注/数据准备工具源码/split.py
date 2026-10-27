"""带标签数据集划分命令行入口；具体文件操作由共用服务完成。"""

import argparse
from pathlib import Path

from dataset_tool.services.labeled_splitter import split_labeled_dataset

SCRIPT_DIR = Path(__file__).resolve().parent


def split_dataset(images_dir=None, labels_dir=None, output_dir=None, train_ratio=0.7,
                  val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """兼容原函数参数；仅复制同名 JPG/TXT 配对，返回划分统计。"""
    result = split_labeled_dataset(
        images_dir or SCRIPT_DIR / "images", labels_dir or SCRIPT_DIR / "labels",
        output_dir or SCRIPT_DIR / "iceice", train_ratio, val_ratio, test_ratio, random_seed,
        progress_callback=lambda event: print("复制进度：{}/{} ({})".format(event["current"], event["total"], event["split"]))
    )
    if result["warning"]:
        print("提示：" + result["warning"])
    if result["missing_labels"]:
        print("警告：跳过 {} 张缺少标签的图片".format(len(result["missing_labels"])))
    print("划分完成：{}，各集合数量：{}".format(result["output_dir"], result["counts"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="按图片/标签配对划分 YOLO 数据集")
    parser.add_argument("--images", default=str(SCRIPT_DIR / "images"), help="图片目录")
    parser.add_argument("--labels", default=str(SCRIPT_DIR / "labels"), help="标签目录")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "iceice"), help="空的输出目录")
    parser.add_argument("--train", type=float, default=0.7, help="训练比例，默认 0.7")
    parser.add_argument("--val", type=float, default=0.2, help="验证比例，默认 0.2")
    parser.add_argument("--test", type=float, default=0.1, help="测试比例，默认 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args(argv)
    try:
        split_dataset(args.images, args.labels, args.output, args.train, args.val, args.test, args.seed)
        return 0
    except Exception as error:
        parser.exit(1, "错误：{}\n".format(error))


if __name__ == "__main__":
    raise SystemExit(main())
