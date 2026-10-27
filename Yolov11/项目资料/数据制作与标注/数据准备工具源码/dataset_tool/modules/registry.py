"""功能模块清单。"""

from .base import ToolModule


def build_modules(video_page, labeled_page, sample_page, unavailable_page):
    return [
        ToolModule("video", "视频抽帧", video_page),
        ToolModule("labeled_split", "图片和标签划分", labeled_page),
        ToolModule("image_sample", "图片抽样划分", sample_page),
        ToolModule("annotation", "标注（未接入）", unavailable_page, False, "标注工具接口预留，当前版本不执行标注。"),
        ToolModule("training", "训练（未接入）", unavailable_page, False, "训练工具接口预留，当前版本不启动训练。"),
    ]
