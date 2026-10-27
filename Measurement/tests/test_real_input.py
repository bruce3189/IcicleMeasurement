import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore, QtTest
from icicle_measure.capture import (CapturePacket, CaptureRunner, LatestFrameBuffer,
                                     PinholeRectifier, RealSenseSource)
from icicle_measure.pose import PoseConfig, PoseDetector
from icicle_measure.simulator import generate
from icicle_measure.io import save_simulation, load_bundle, write_json, depth_quality, save_input_bundle
from icicle_measure.core import measure_frame
from icicle_measure.gui import DepthTerrainCanvas, MainWindow, configure_font
from icicle_measure.live_gui import CameraDialog
from icicle_measure.render import overlay
from tests.helpers import temporary_directory


def packet(sequence=1):
    s = generate('clean')
    s.frame.metadata.update(input_kind='camera_test', sequence=sequence,
                            captured_at='2026-10-25T12:00:00+00:00', timestamp_ms=sequence * 33.3)
    return CapturePacket(s.frame, s.intrinsics, sequence, sequence * 33.3, s.frame.metadata['captured_at'])


class InputTests(unittest.TestCase):
    def test_raw_scale_and_replay_provenance(self):
        with temporary_directory() as directory:
            s = generate('clean')
            manifest = save_simulation(directory, s)
            root = Path(directory)
            scale = .00025
            raw = np.rint(s.frame.depth_m / scale).astype(np.uint16)
            cv2.imencode('.png', raw)[1].tofile(str(root / 'depth.png'))
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            meta.update(depth='depth.png', depth_unit='raw', depth_scale_m_per_unit=scale,
                        intrinsics_image_size=[320, 240])
            write_json(manifest, meta)
            frame, dets, intrinsics = load_bundle(manifest)
            np.testing.assert_allclose(frame.depth_m, s.frame.depth_m, atol=scale / 2)
            result = measure_frame(frame, dets, intrinsics)
            self.assertEqual(result.objects[0].status, 'accepted')
            self.assertEqual(result.source['depth_scale_m_per_unit'], scale)
            replay = save_input_bundle(root / 'replay', frame, dets, intrinsics)
            reread = load_bundle(replay)[0]
            np.testing.assert_array_equal(reread.depth_m, frame.depth_m)
            self.assertEqual(reread.metadata['input_kind'], 'simulated')
            for value in (0, -1, float('inf'), True):
                write_json(manifest, dict(meta, depth_scale_m_per_unit=value)) if np.isfinite(value) else manifest.write_text(json.dumps(dict(meta, depth_scale_m_per_unit=value)), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_bundle(manifest)
            write_json(manifest, dict(meta, depth_unit='mm'))
            with self.assertRaisesRegex(ValueError, '重复换算'):
                load_bundle(manifest)

    def test_preview_shape_intrinsics_and_coverage(self):
        with temporary_directory() as directory:
            manifest = save_simulation(directory, generate('clean'))
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            root = Path(directory)
            for array in (np.zeros((240, 320), np.uint8), np.zeros((240, 320, 3), np.uint16),
                          np.zeros((240, 320), complex), np.zeros((10, 10), float)):
                np.save(str(root / 'depth.npy'), array)
                with self.assertRaises(ValueError):
                    load_bundle(manifest)
            np.save(str(root / 'depth.npy'), np.zeros((240, 320), float))
            frame, _, _ = load_bundle(manifest)
            self.assertEqual(depth_quality(frame)['valid_fraction'], 0)
            write_json(manifest, dict(meta, intrinsics_image_size=[640, 480]))
            with self.assertRaisesRegex(ValueError, '内参标定尺寸'):
                load_bundle(manifest)

    def test_missing_tip_retains_root_color_and_invalid_boxes_do_not_crash(self):
        s = generate('clean')
        detection = replace(s.detections[0], keypoints=[None, (160., 40.)])
        image = overlay(s.frame, [detection], boxes=False, distances=False)
        self.assertEqual(tuple(image[40, 160]), (70, 225, 240))
        for box in ((1, 2, 3), (0, 0, 1e100, 1e100)):
            overlay(s.frame, [replace(detection, bbox=box)])


class CameraTests(unittest.TestCase):
    def test_size_one_mailbox(self):
        mailbox = LatestFrameBuffer()
        for n in range(1, 20):
            mailbox.put(packet(n))
        self.assertEqual(mailbox.take().sequence, 19)
        self.assertEqual(mailbox.overwritten, 18)
        self.assertIsNone(mailbox.take())

    def test_sdk_alignment_device_scale_and_lifecycle(self):
        rs = Mock()
        rs.context.return_value.query_devices.return_value = ['test-device']
        raw = np.array([[0, 5000], [6000, 6500]], np.uint16)
        rgb = np.full((2, 2, 3), 100, np.uint8)
        ci = SimpleNamespace(fx=200., fy=210., ppx=.5, ppy=.5, coeffs=[0.] * 5, model='none')
        color, depth = Mock(), Mock()
        color.get_data.return_value, depth.get_data.return_value = rgb, raw
        color.profile.as_video_stream_profile.return_value.intrinsics = ci
        color.get_frame_number.return_value, color.get_timestamp.return_value = 23, 1000.
        frames = Mock()
        frames.get_color_frame.return_value, frames.get_depth_frame.return_value = color, depth
        rs.pipeline.return_value.try_wait_for_frames.return_value = True, frames
        rs.align.return_value.process.return_value = frames
        device = rs.pipeline.return_value.start.return_value.get_device.return_value
        device.first_depth_sensor.return_value.get_depth_scale.return_value = .00025
        device.get_info.return_value = 'test-device'
        source = RealSenseSource(sdk=rs)
        source.open()
        observed = source.read()
        np.testing.assert_array_equal(observed.frame.depth_m, raw * .00025)
        self.assertEqual(observed.sequence, 23)
        self.assertEqual(observed.intrinsics.fx, 200.)
        self.assertTrue(observed.frame.metadata['aligned_to_rgb'])
        rs.align.assert_called_once_with(rs.stream.color)
        source.close()
        rs.pipeline.return_value.stop.assert_called_once()

    def test_rectification_preserves_depth_edges(self):
        p = packet()
        z = p.frame.depth_m.copy()
        z[:, :160], z[:, 160:] = 1., 3.
        rectifier = PinholeRectifier()
        rgb, z2 = rectifier.apply(p.frame.rgb, z, p.intrinsics, [.2, 0, 0, 0, 0], 'brown_conrady')
        self.assertEqual(rgb.shape, p.frame.rgb.shape)
        self.assertLessEqual(set(np.unique(z2)), {0., 1., 3.})
        with self.assertRaises(ValueError):
            rectifier.apply(p.frame.rgb, z, p.intrinsics, [.1, 0, 0, 0, 0], 'inverse_brown_conrady')

    def test_timeout_error_and_stop_release_source(self):
        source = Mock()
        source.read.return_value = None
        runner = CaptureRunner(lambda: source, timeout_s=.01)
        runner.start(); runner.join()
        self.assertFalse(runner.is_alive())
        self.assertIn('未提供完整', runner.error)
        source.close.assert_called_once()
        source = Mock()
        source.read.side_effect = RuntimeError('disconnected')
        runner = CaptureRunner(lambda: source)
        runner.start(); runner.join()
        self.assertEqual(runner.error, 'disconnected')
        source.close.assert_called_once()


class PoseTests(unittest.TestCase):
    def test_instance_order_rgb_conversion_and_missing_keypoints(self):
        with temporary_directory() as directory:
            path = Path(directory) / 'fake.pt'; path.touch()
            config = PoseConfig(str(path))
            tensor = lambda data: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.asarray(data)))
            result = SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor([[1, 2, 10, 20], [30, 40, 50, 60]])),
                keypoints=SimpleNamespace(xy=tensor([[[3, 4], [5, 6]], [[31, 41], [42, 52]]]),
                                          conf=tensor([[.9, .1], [.8, .9]])))
            model = Mock(); model.task = 'pose'; model.model.kpt_shape = [2, 3]
            model.predict.return_value = [result]
            detector = PoseDetector(config, lambda _: model)
            p = packet(42)
            detections = detector.detect(p.frame, 42)
            self.assertEqual([d.id for d in detections], ['42:1', '42:2'])
            self.assertEqual(detections[0].keypoints, [(3., 4.), None])
            self.assertEqual(detections[1].bbox, (30., 40., 50., 60.))
            np.testing.assert_array_equal(model.predict.call_args.kwargs['source'], p.frame.rgb[:, :, ::-1])
            model.model.kpt_shape = [17, 3]
            with self.assertRaises(ValueError):
                PoseDetector(config, lambda _: model)


class LiveGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_font(cls.app)

    def test_cloud_occlusion_and_category_visibility(self):
        s = generate('edge')
        result = measure_frame(s.frame, s.detections, s.intrinsics)
        canvas = DepthTerrainCanvas(); canvas.resize(700, 480)
        canvas.set_data(result.objects[0].endpoints[0], s.frame.depth_m, s.detections[0], s.intrinsics)
        canvas.radius = 10; canvas.rotation = np.array([1., 0, 0, 0])
        canvas.vertices = np.array([[6., 0, 0], [5., 0, 10]])
        canvas.vertex_categories = np.array([3, 2])
        canvas.sample_categories = canvas.vertex_categories.copy()
        canvas.sample_pixel_ids = np.array([1, 2]); canvas.sample_depths = np.array([1., 2.])
        canvas.visible_mask = np.ones(2, bool)
        canvas.show(); self.app.processEvents()
        img = QtGui.QImage(canvas.size(), QtGui.QImage.Format_ARGB32); canvas.render(img)
        pt = canvas._proj_screen[-1]
        color = img.pixelColor(int(round(pt[0])), int(round(pt[1])))
        self.assertGreater(color.blue(), color.green(), 'front blue sample must obscure far green sample')
        coords = canvas.vertices.copy()
        canvas.set_category_visible(2, False); self.app.processEvents()
        np.testing.assert_array_equal(coords, canvas.vertices)
        canvas.close()

    def test_grab_and_continuous_use_latest_packet_without_reusing_old_result(self):
        dialog = CameraDialog(lambda: True)
        dialog.timer.stop()
        dialog.runner = Mock()
        dialog.runner.is_alive.return_value = True
        dialog.runner.stop_event.is_set.return_value = False
        dialog.runner.buffer.overwritten = 2
        dialog.runner.buffer.take.return_value = packet(3)
        captured = []
        dialog.measure_requested.connect(lambda p, c: captured.append(p.sequence))
        dialog.poll()
        with patch.object(PoseConfig, 'validate'):
            dialog.submit()
            self.assertEqual(captured, [3])
            dialog.continuous.setChecked(True)
            dialog.next_inference = 0
            dialog.poll()
            self.assertEqual(captured, [3])
            dialog.runner.buffer.take.return_value = packet(7)
            dialog.poll()
            self.assertEqual(captured, [3, 7])
        dialog.runner = None; dialog.close()

    def test_pose_worker_snapshot_has_matching_frame_and_export_source(self):
        window = MainWindow()
        p = packet(14)
        detector = Mock(); detector.config = PoseConfig('injected.pt')
        detector.detect.return_value = generate('clean').detections
        window.pose_detector = detector
        window.start_pose_measurement(p, detector.config)
        deadline = time.monotonic() + 5
        while window.thread is not None and time.monotonic() < deadline:
            self.app.processEvents(); QtTest.QTest.qWait(5)
        self.assertIsNone(window.thread)
        self.assertIs(window.frame, p.frame)
        self.assertEqual(window.result.source['sequence'], 14)
        self.assertEqual(window.result.objects[0].status, 'accepted')
        with temporary_directory() as directory:
            window.export_to(directory)
            saved = json.loads((Path(directory) / 'measurement.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['source']['sequence'], 14)
            self.assertEqual(saved['objects'][0]['distance_m'], window.result.objects[0].distance_m)
        window.close()


if __name__ == '__main__':
    unittest.main()
