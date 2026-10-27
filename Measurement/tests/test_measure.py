import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import cv2
from icicle_measure import Config, Intrinsics, Frame, Detection, Strategies, measure_frame
from icicle_measure.core import (deproject, shared_membership, otsu_split,
                                 select_target, confidence_scores)
from icicle_measure.models import WindowRecord
from icicle_measure.simulator import generate
from icicle_measure.io import (load_bundle, save_simulation, export_results, load_detections,
                              write_json, load_files)
from tests.helpers import temporary_directory


class MeasurementTests(unittest.TestCase):
    def run_scene(self, name, seed=42, **kwargs):
        s = generate(name, seed)
        return s, measure_frame(s.frame, s.detections, s.intrinsics, **kwargs)

    def test_default_sampling_requires_large_clean_clusters(self):
        cfg = Config()
        self.assertGreaterEqual(cfg.min_samples, 64)
        self.assertGreaterEqual(cfg.full_samples, 100)
        self.assertGreaterEqual(max(cfg.radii), 60)
        self.assertEqual(cfg.bbox_padding, 10)
        s, result = self.run_scene('clean')
        self.assertEqual(result.objects[0].status, 'accepted')
        for ep in result.objects[0].endpoints:
            record = ep.history[-1]
            self.assertGreaterEqual(record.near_count, 80)
            self.assertGreaterEqual(record.far_count, 80)
            self.assertGreaterEqual(record.target_count, 80)
            self.assertGreaterEqual(record.radius, 8)

    def test_deprojection_original_ray_and_distance(self):
        self.assertEqual(deproject((100, 200), 2, Intrinsics(200, 400, 0, 0)), (1., 1., 2.))
        s, result = self.run_scene('clean')
        obj = result.objects[0]
        self.assertEqual(obj.status, 'accepted')
        self.assertAlmostEqual(obj.distance_m, s.truth[obj.id]['distance_m'], places=12)
        for i, ep in enumerate(obj.endpoints):
            np.testing.assert_allclose(ep.xyz_m, s.truth[obj.id]['xyz_m'][i], atol=1e-12)
            self.assertEqual(ep.pixel, s.detections[0].keypoints[i])
            self.assertGreaterEqual(len(ep.history), 3)

    def test_arithmetic_mean_uses_selected_cleaned_samples(self):
        s, result = self.run_scene('outliers')
        for ep in result.objects[0].endpoints:
            self.assertEqual(ep.status, 'accepted')
            ids = ep.history[-1].target_ids
            self.assertAlmostEqual(ep.depth_m, np.mean(s.frame.depth_m.ravel()[ids]), places=12)

    def test_holes_and_wrong_nonzero_depth(self):
        for scenario in ('holes', 'background_endpoint'):
            s, result = self.run_scene(scenario)
            self.assertEqual(result.objects[0].status, 'accepted')
            for i, ep in enumerate(result.objects[0].endpoints):
                self.assertLess(abs(ep.depth_m - s.truth['ice-1']['depth_m'][i]), .01)

    def test_unmeasurable_scenes_reject(self):
        for scenario in ('sparse', 'overlap', 'background', 'empty', 'drift', 'membership', 'interference'):
            with self.subTest(scenario=scenario):
                _, result = self.run_scene(scenario)
                self.assertEqual(result.objects[0].status, 'rejected')
                self.assertIsNone(result.objects[0].distance_m)
                self.assertTrue(result.objects[0].reason)
                self.assertTrue(all(ep.depth_m is None for ep in result.objects[0].endpoints))

    def test_noise_benchmark_30_seeds(self):
        accepted = 0
        for seed in range(30):
            s, result = self.run_scene('noisy', seed)
            obj = result.objects[0]
            if obj.status == 'accepted':
                accepted += 1
                self.assertLess(abs(obj.distance_m - s.truth[obj.id]['distance_m']), .02)
                for index, ep in enumerate(obj.endpoints):
                    self.assertLess(abs(ep.depth_m - s.truth[obj.id]['depth_m'][index]), .01)
        self.assertGreaterEqual(accepted, 27)

    def test_millimetre_quantization(self):
        s = generate('clean')
        s.frame.depth_m = np.round(s.frame.depth_m * 1000) / 1000
        result = measure_frame(s.frame, s.detections, s.intrinsics)
        self.assertLess(abs(result.objects[0].distance_m - s.truth['ice-1']['distance_m']), .002)

    def test_multi_and_image_edge(self):
        _, result = self.run_scene('multi')
        self.assertEqual([o.status for o in result.objects], ['accepted', 'accepted', 'skipped'])
        # 图像边缘：检测框短边仅 17 px，窗口半径被截断到 8 px 后仍达不到
        # 连续扩窗判定要求，按 README「到达短边上限仍未通过立即拒绝」拒绝。
        _, result = self.run_scene('edge')
        self.assertEqual(result.objects[0].status, 'rejected')
        self.assertTrue(all(ep.status == 'rejected' for ep in result.objects[0].endpoints))
        self.assertTrue(all('短边搜索上限' in (ep.reason or '') for ep in result.objects[0].endpoints))

    def test_shared_membership_ignores_new_area(self):
        a = WindowRecord(3, (0, 0, 3, 3), valid_ids=[1, 2, 3], target_ids=[1, 2])
        b = WindowRecord(5, (0, 0, 5, 5), valid_ids=[1, 2, 3, 4, 5], target_ids=[1, 2, 4, 5])
        self.assertEqual(shared_membership(a, b), 1)
        b.target_ids = [2, 3, 4, 5]
        self.assertAlmostEqual(shared_membership(a, b), 1 / 3)

    def test_membership_change_resets_consecutive_comparisons(self):
        s = generate('clean')
        calls = [0]

        def alternating(ids, depths, labels, shape, detection, config):
            target, reason = select_target(ids, depths, labels, shape, detection, config)
            calls[0] += 1
            xs = ids % shape[1]
            target &= (xs <= 160) if calls[0] % 2 else (xs > 160)
            return target, reason

        result = measure_frame(s.frame, s.detections, s.intrinsics, strategies=Strategies(selector=alternating))
        self.assertEqual(result.objects[0].status, 'rejected')
        for ep in result.objects[0].endpoints:
            self.assertTrue(any(r.membership == 0 for r in ep.history))

    def test_clipped_window_does_not_count_twice(self):
        s = generate('clean')
        d = Detection('tiny', (158, 39, 162, 44), [(160, 40), (161, 43)])
        result = measure_frame(s.frame, [d], s.intrinsics, Config(radii=(20, 30, 40), bbox_padding=0))
        for ep in result.objects[0].endpoints:
            self.assertEqual(len(ep.history), 1)
            self.assertEqual(ep.status, 'rejected')

    def test_min_score_not_average(self):
        scores = confidence_scores(100, 2, 0, .01, 1, Config())
        self.assertAlmostEqual(min(scores.values()), 2 / 100)

    def test_only_both_accepted_produce_distance(self):
        s = generate('clean')
        s.frame.depth_m[:80] = 0
        obj = measure_frame(s.frame, s.detections, s.intrinsics).objects[0]
        self.assertEqual(obj.endpoints[0].status, 'accepted')
        self.assertEqual(obj.endpoints[1].status, 'rejected')
        self.assertIsNone(obj.distance_m)

    def test_invalid_inputs_and_config(self):
        s = generate()
        with self.assertRaises(ValueError):
            measure_frame(Frame(s.frame.rgb, np.zeros((2, 2))), s.detections, s.intrinsics)
        with self.assertRaises(ValueError):
            measure_frame(s.frame, s.detections, Intrinsics(0, 1, 0, 0))
        for data in ({'radii': [3, 3, 8]}, {'radii': [3.5, 5, 8]}, {'confidence_min': 1.1},
                     {'min_samples': -1}, {'center_abs_m': float('nan')}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                Config.from_dict(data)
        bad = Detection('bad', (0, 0, 20, 20), [(-1., 0), (10., 10.)])
        self.assertEqual(measure_frame(s.frame, [bad], s.intrinsics).objects[0].status, 'skipped')

    def test_cancel_before_and_during_estimation(self):
        s = generate()
        result = measure_frame(s.frame, s.detections, s.intrinsics, cancel=lambda: True)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.objects, [])
        calls = [0]
        def cancel():
            calls[0] += 1
            return calls[0] >= 3
        result = measure_frame(s.frame, s.detections, s.intrinsics, cancel=cancel)
        self.assertTrue(result.cancelled)
        self.assertIsNone(result.objects[0].distance_m)


class AdapterTests(unittest.TestCase):
    def test_bundle_roundtrip_and_export(self):
        s = generate('multi')
        with temporary_directory() as directory:
            root = Path(directory)
            manifest = save_simulation(root / 'input', s)
            frame, detections, intrinsics = load_bundle(manifest)
            np.testing.assert_array_equal(frame.rgb, s.frame.rgb)
            np.testing.assert_array_equal(frame.depth_m, s.frame.depth_m)
            result = measure_frame(frame, detections, intrinsics)
            export_results(root / 'output', frame, detections, result)
            saved = json.loads((root / 'output/measurement.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['objects'][0]['distance_m'], result.objects[0].distance_m)
            self.assertTrue((root / 'output/overlay.png').is_file())
            self.assertIn('distance_m', (root / 'output/measurement.csv').read_text(encoding='utf-8-sig'))

    def test_explicit_mm_unit_and_metadata_errors(self):
        s = generate('clean')
        with temporary_directory() as directory:
            root = Path(directory)
            manifest = save_simulation(root, s)
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            np.save(str(root / 'depth.npy'), s.frame.depth_m * 1000)
            meta['depth_unit'] = 'mm'
            write_json(manifest, meta)
            frame, _, _ = load_bundle(manifest)
            np.testing.assert_allclose(frame.depth_m, s.frame.depth_m)
            for key, value in [('depth_unit', 'cm'), ('aligned_to_rgb', False), ('depth_kind', 'radial')]:
                broken = dict(meta, **{key: value})
                write_json(manifest, broken)
                with self.assertRaises(ValueError):
                    load_bundle(manifest)

    def test_yolo_adapter_visibility_and_order(self):
        with temporary_directory() as directory:
            path = Path(directory) / 'labels.txt'
            path.write_text('0 .5 .5 .2 .8 .5 .9 .95 .5 .1 .9 .8\n0 .3 .5 .1 .8 .3 .9 0 .3 .1 .8\n')
            detections = load_detections(path, 100, 200)
            self.assertEqual(detections[0].keypoints, [(50., 180.), (50., 20.)])
            self.assertIsNone(detections[1].keypoints[0])

    def test_16bit_png_depth(self):
        with temporary_directory() as directory:
            root = Path(directory)
            s = generate('clean')
            manifest = save_simulation(root, s)
            ok, encoded = cv2.imencode('.png', np.round(s.frame.depth_m * 1000).astype(np.uint16))
            self.assertTrue(ok)
            encoded.tofile(str(root / 'depth.png'))
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            meta.update(depth='depth.png', depth_unit='mm')
            write_json(manifest, meta)
            frame, detections, intrinsics = load_bundle(manifest)
            obj = measure_frame(frame, detections, intrinsics).objects[0]
            self.assertEqual(obj.status, 'accepted')
            self.assertLess(abs(obj.distance_m - s.truth['ice-1']['distance_m']), .002)


if __name__ == '__main__':
    unittest.main()
