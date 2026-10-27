"""OpenCV 视频抽帧服务。"""

import math
from pathlib import Path

import cv2

from .common import check_cancel, emit_progress, prepare_empty_output


def extract_video(video_path, output_dir, target_fps=2.0, progress_callback=None, cancel_callback=None,
                  name_prefix=None, allow_existing_output=False):
    """按目标 FPS 近似采样视频帧，返回视频信息和图片数量。"""
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError("视频文件不存在：{}".format(source))
    if not isinstance(target_fps, (int, float)) or not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("目标 FPS 必须是大于 0 的有限数值")
    output = Path(output_dir)
    if allow_existing_output:
        if output.exists() and not output.is_dir():
            raise NotADirectoryError("输出路径不是目录：{}".format(output))
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = prepare_empty_output(output)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV 无法打开视频：{}".format(source))
    saved = 0
    try:
        orientation_meta = float(capture.get(cv2.CAP_PROP_ORIENTATION_META)) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0.0
        orientation_auto = False
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            orientation_auto = bool(capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("视频帧率无效：{}".format(source))
        if total_frames <= 0:
            raise ValueError("视频没有可读取的帧：{}".format(source))
        interval = max(1, int(source_fps / float(target_fps)))
        expected = int(math.ceil(total_frames / float(interval)))
        prefix = name_prefix or source.stem
        frame_index = 0
        while True:
            check_cancel(cancel_callback)
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                target = output / "{}_{:05d}.jpg".format(prefix, saved + 1)
                if not cv2.imwrite(str(target), frame):
                    raise OSError("无法写入图片：{}".format(target))
                saved += 1
                emit_progress(progress_callback, current=saved, total=expected, frame=frame_index, path=str(target))
            frame_index += 1
        if saved == 0:
            raise ValueError("视频未能解码出任何图片：{}".format(source))
        return {"image_count": saved, "source_fps": source_fps, "total_frames": total_frames,
                "width": width, "height": height, "frame_interval": interval,
                "duration": total_frames / source_fps,
                "orientation_degrees": orientation_meta,
                "orientation_auto": orientation_auto}
    finally:
        capture.release()
