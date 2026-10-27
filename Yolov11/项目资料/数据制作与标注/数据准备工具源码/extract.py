"""视频抽帧命令行入口；GUI 与本脚本共用 dataset_tool.services.video_extractor。"""

import argparse
import math
from pathlib import Path

from dataset_tool.services.common import prepare_empty_output
from dataset_tool.services.video_extractor import extract_video


def extract_frames(video_path, output_dir, fps_to_extract, video_name):
    """兼容旧函数名，处理一个视频并返回抽取数量。"""
    result = extract_video(video_path, output_dir, fps_to_extract, name_prefix=video_name)
    return result["image_count"]


def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从视频中按目标 FPS 提取帧")
    parser.add_argument("--fps", type=float, default=2.0, help="每秒提取帧数，默认 2")
    parser.add_argument("--video_dir", type=Path, default=script_dir / "videos", help="视频文件目录")
    parser.add_argument("--output_dir", type=Path, default=script_dir / "images", help="输出图片目录")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps 必须是大于 0 的有限数值")
    if not args.video_dir.is_dir():
        parser.error("视频目录不存在：{}".format(args.video_dir))
    videos = sorted(path for path in args.video_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
    if not videos:
        parser.error("目录中没有找到 MP4 视频：{}".format(args.video_dir))
    try:
        # 整批任务只检查一次目录。每个视频仍使用共用服务，按视频名前缀写入文件。
        output = prepare_empty_output(args.output_dir)
        total = 0
        for video in videos:
            info = extract_video(video, output, args.fps, name_prefix=video.stem, allow_existing_output=True)
            total += info["image_count"]
            print("{}：提取 {} 张图片".format(video.name, info["image_count"]))
        print("完成，共提取 {} 张图片：{}".format(total, output))
        return 0
    except Exception as error:
        parser.exit(1, "错误：{}\n".format(error))


if __name__ == "__main__":
    raise SystemExit(main())
