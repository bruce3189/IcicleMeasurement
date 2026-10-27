"""RGB-D acquisition. Importing this module never imports a camera SDK.

Packets own their arrays. A size-one mailbox drops stale preview frames; the
measurement worker holds its own packet, so detections/depth/results cannot
accidentally refer to different frames.
"""
import importlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import cv2
import numpy as np
from .models import Frame, Intrinsics


@dataclass(frozen=True)
class CapturePacket:
    frame: Frame
    intrinsics: Intrinsics
    sequence: int
    timestamp_ms: float
    captured_at: str


class LatestFrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._packet = None
        self._unread = False
        self.overwritten = 0

    def put(self, packet):
        with self._lock:
            if self._unread:
                self.overwritten += 1
            self._packet, self._unread = packet, True

    def take(self):
        with self._lock:
            if not self._unread:
                return None
            self._unread = False
            return self._packet


class PinholeRectifier:
    """Rectify aligned RGB and Z with the same map; never interpolate Z edges."""
    def __init__(self):
        self._key = self._maps = None

    def apply(self, rgb, depth_m, intrinsics, coefficients, model='none'):
        coeffs = tuple(float(x) for x in coefficients)
        name = str(model).lower().split('.')[-1]
        if name == 'none' or not any(abs(x) > 1e-12 for x in coeffs):
            return rgb, depth_m
        if name not in ('brown_conrady', 'modified_brown_conrady'):
            raise ValueError('不支持该彩色畸变模型：%s；不能将其当成已去畸变针孔图像' % model)
        h, w = depth_m.shape
        key = (w, h, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, coeffs)
        if key != self._key:
            k = np.asarray([[intrinsics.fx, 0, intrinsics.cx],
                            [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], dtype=float)
            self._maps = cv2.initUndistortRectifyMap(k, np.asarray(coeffs), None, k,
                                                   (w, h), cv2.CV_32FC1)
            self._key = key
        return (cv2.remap(rgb, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
                cv2.remap(depth_m, *self._maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT))


class RealSenseSource:
    def __init__(self, width=640, height=480, fps=30, sdk=None):
        self.width, self.height, self.fps = width, height, fps
        self.sdk = sdk
        self.pipeline = None
        self.started = False
        self.rectifier = PinholeRectifier()

    def open(self):
        try:
            rs = self.sdk or importlib.import_module('pyrealsense2')
        except ImportError as exc:
            raise RuntimeError('未安装 pyrealsense2。Python 3.9 请使用 requirements-camera-py39.txt；离线功能仍可使用。') from exc
        self.sdk = rs
        if not len(rs.context().query_devices()):
            raise RuntimeError('未检测到 RealSense 深度相机。请连接 D435i 后重试；当前离线功能可正常使用。')
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        try:
            profile = self.pipeline.start(config)
            self.started = True
            device = profile.get_device()
            self.scale = float(device.first_depth_sensor().get_depth_scale())
            if not np.isfinite(self.scale) or self.scale <= 0:
                raise RuntimeError('相机返回无效 depth scale')
            self.serial = device.get_info(rs.camera_info.serial_number)
            self.align = rs.align(rs.stream.color)
        except Exception:
            self.close()
            raise

    def read(self, timeout_ms=250) -> Optional[CapturePacket]:
        ok, frames = self.pipeline.try_wait_for_frames(timeout_ms)
        if not ok:
            return None
        frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        rgb = np.asanyarray(color.get_data()).copy()
        z = np.asanyarray(depth.get_data()).astype(np.float64) * self.scale
        ci = color.profile.as_video_stream_profile().intrinsics
        intrinsics = Intrinsics(float(ci.fx), float(ci.fy), float(ci.ppx), float(ci.ppy))
        intrinsics.validate()
        rgb, z = self.rectifier.apply(rgb, z, intrinsics, ci.coeffs, ci.model)
        sequence, timestamp = int(color.get_frame_number()), float(color.get_timestamp())
        captured = datetime.now(timezone.utc).isoformat()
        frame = Frame(rgb, z, metadata={
            'input_kind': 'realsense', 'serial': self.serial,
            'sequence': sequence, 'timestamp_ms': timestamp, 'captured_at': captured,
            'depth_unit': 'raw', 'depth_scale_m_per_unit': self.scale,
            'aligned_to_rgb': True, 'undistorted': True,
            'intrinsics_image_size': [rgb.shape[1], rgb.shape[0]],
            'depth_filters': 'none; no hole filling'})
        frame.validate()
        return CapturePacket(frame, intrinsics, sequence, timestamp, captured)

    def close(self):
        pipeline, self.pipeline = self.pipeline, None
        started, self.started = self.started, False
        if pipeline is not None and started:
            pipeline.stop()


class CaptureRunner:
    def __init__(self, source_factory=RealSenseSource, timeout_s=5.0):
        self.source_factory = source_factory
        self.timeout_s = timeout_s
        self.buffer = LatestFrameBuffer()
        self.stop_event = threading.Event()
        self.error = ''
        self.thread = None

    def start(self):
        if self.is_alive():
            raise RuntimeError('采集已启动')
        self.stop_event.clear()
        self.error = ''
        self.thread = threading.Thread(target=self._run, name='rgbd-capture', daemon=True)
        self.thread.start()

    def _run(self):
        source = None
        try:
            source = self.source_factory()
            source.open()
            last_frame = time.monotonic()
            while not self.stop_event.is_set():
                packet = source.read(timeout_ms=250)
                if packet is not None:
                    self.buffer.put(packet)
                    last_frame = time.monotonic()
                elif time.monotonic() - last_frame > self.timeout_s:
                    raise RuntimeError('相机超过 %.1f 秒未提供完整 RGB-D 帧，请检查连接' % self.timeout_s)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    self.error = self.error or '关闭相机失败：' + str(exc)

    def stop(self):
        self.stop_event.set()

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def join(self, timeout=2.0):
        if self.thread is not None:
            self.thread.join(timeout)
