"""Optional two-keypoint YOLO adapter; instance indices are never reordered."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .models import Detection


@dataclass(frozen=True)
class PoseConfig:
    weights: str
    box_confidence: float = .25
    keypoint_confidence: float = .3
    image_size: int = 640
    reversed_order: bool = False

    def validate(self):
        if not Path(self.weights).is_file():
            raise ValueError('请选择已有的两关键点 YOLO Pose 权重文件')
        if not 0 <= self.box_confidence <= 1 or not 0 <= self.keypoint_confidence <= 1:
            raise ValueError('模型置信度阈值须在 0..1')
        if self.image_size < 32:
            raise ValueError('模型输入边长须至少为 32')


class PoseDetector:
    def __init__(self, config, model_factory=None):
        config.validate()
        self.config = config
        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError('当前 Python 未安装 Ultralytics；离线标注导入仍可使用') from exc
            model_factory = YOLO
        self.model = model_factory(config.weights)
        shape = getattr(self.model.model, 'kpt_shape', None)
        if self.model.task != 'pose' or shape is None or list(shape) != [2, 3]:
            raise ValueError('权重必须为两个关键点且含置信度的 Pose 模型，kpt_shape=[2,3]')

    def detect(self, frame, sequence=0):
        # Ultralytics treats ndarray input as BGR. Frame.rgb is explicitly RGB.
        result = self.model.predict(source=frame.rgb[:, :, ::-1].copy(),
                                    imgsz=self.config.image_size, conf=self.config.box_confidence,
                                    device='cpu', verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        if not len(boxes):
            return []
        pixels = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf
        if conf is None:
            raise ValueError('Pose 输出缺少端点置信度')
        conf = conf.cpu().numpy()
        if len(boxes) != len(pixels) or pixels.shape[1:] != (2, 2) or conf.shape != (len(boxes), 2):
            raise ValueError('框与两端点输出数量不匹配')
        order = (1, 0) if self.config.reversed_order else (0, 1)
        detections = []
        for i, box in enumerate(boxes):
            points = [tuple(map(float, pixels[i, j])) if np.isfinite(conf[i, j]) and
                      conf[i, j] > self.config.keypoint_confidence else None for j in order]
            detections.append(Detection('%s:%d' % (sequence, i + 1), tuple(map(float, box)), points))
        return detections


def default_weights():
    path = Path(__file__).resolve().parents[1] / 'outputs/ice_pose_runs/yolo11n_pose_78_2kpt/weights/best.pt'
    return str(path) if path.is_file() else ''
