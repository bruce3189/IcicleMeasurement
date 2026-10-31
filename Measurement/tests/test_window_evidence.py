import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from PyQt5 import QtWidgets
from icicle_measure.models import Config, Detection, WindowRecord
from icicle_measure.simulator import generate
from icicle_measure.core import measure_frame, select_target, otsu_split, shared_membership
from icicle_measure.evidence import (subgroup_evidence, plane_evidence,
                                     growth_evidence, coverage_evidence)
from icicle_measure.gui import DiagnosticsPlot, configure_font
from icicle_measure.io import export_results
from tests.helpers import temporary_directory


# Historical evidence contract; new default behavior is covered separately by
# test_near_cluster_simple.py. Do not silently apply old 80-point/anchor gates
# to the current near-first algorithm.
from dataclasses import dataclass
_CurrentConfig, _measure_current = Config, measure_frame
@dataclass
class Config(_CurrentConfig):
    endpoint_depth_mode: str = 'near_cluster_mean'
def measure_frame(*args, **kwargs):
    if len(args) < 4 and kwargs.get('config') is None:
        kwargs['config'] = Config()
    return _measure_current(*args, **kwargs)

class WindowEvidenceTests(unittest.TestCase):
    def test_slope_bias_is_rejected_without_using_model_prediction(self):
        s = generate('slope_bias')
        obj = measure_frame(s.frame, s.detections, s.intrinsics, Config(endpoint_depth_mode='strict_local')).objects[0]
        self.assertEqual(obj.status, 'rejected')
        self.assertIsNone(obj.distance_m)
        for i, ep in enumerate(obj.endpoints):
            self.assertIsNone(ep.depth_m)
            last = ep.history[-1]
            self.assertTrue(last.plane['reliable'])
            self.assertFalse(last.checks['representative'])
            self.assertAlmostEqual(last.plane['predicted_depth_m'], s.truth[obj.id]['depth_m'][i], places=10)
            self.assertGreater(last.plane['bias_m'], .005)
            self.assertIn('梯度偏差', ep.reason)

    def test_touching_surfaces_do_not_produce_valid_lengths_across_seeds(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                s = generate('touching_surfaces', seed)
                obj = measure_frame(s.frame, s.detections, s.intrinsics).objects[0]
                self.assertEqual(obj.status, 'rejected')
                self.assertIsNone(obj.distance_m)
                self.assertTrue(any(r.subgroups.get('ambiguous') for ep in obj.endpoints for r in ep.history))

    def test_anchor_selects_local_component_instead_of_larger_remote_one(self):
        width = 40
        ys, xs = np.mgrid[:40, :40]
        ids = (ys * width + xs).ravel()
        local = ((xs >= 9) & (xs <= 11) & (ys >= 8) & (ys <= 12)).ravel()
        remote = ((xs >= 9) & (xs <= 11) & (ys >= 20) & (ys <= 35)).ravel()
        depths, labels = np.full(len(ids), 2.4), np.full(len(ids), 2, np.uint8)
        labels[local | remote] = 1
        depths[local], depths[remote] = 1.5, 1.4
        det = Detection('one', (0, 0, 39, 39), [(10., 10.), (10., 30.)])
        cfg = Config(endpoint_depth_mode='strict_local')
        target, reason = select_target(ids, depths, labels, (40, width), det, cfg, endpoint_pixel=det.keypoints[0])
        self.assertEqual(reason, '')
        np.testing.assert_array_equal(target, local)
        labels[local] = 2
        target, reason = select_target(ids, depths, labels, (40, width), det, cfg, endpoint_pixel=det.keypoints[0])
        self.assertFalse(target.any())
        self.assertIn('锚区', reason)

    def test_secondary_split_does_not_label_flat_gaussian_as_two_surfaces(self):
        rng = np.random.default_rng(8)
        cfg = Config()
        flat = 1.5 + rng.normal(0, .003, 200)
        self.assertFalse(subgroup_evidence(flat, cfg, otsu_split)['ambiguous'])
        mixed = np.r_[np.full(100, 1.5), np.full(100, 1.54)]
        self.assertTrue(subgroup_evidence(mixed, cfg, otsu_split)['ambiguous'])
        self.assertFalse(subgroup_evidence(mixed, replace(cfg, subgroup_min_samples=101), otsu_split)['ambiguous'])

    def test_reclassified_old_pixels_are_not_new_evidence(self):
        prev = WindowRecord(3, (0, 0, 20, 1), target_count=5, far_count=15,
                            valid_ids=list(range(20)), target_ids=list(range(5)))
        ids = np.arange(40)
        target, far = ids < 15, ids >= 15
        z = np.where(target, 1.5, 2.4)
        growth = growth_evidence(prev, ids, target, far, z, Config())
        self.assertEqual(growth['new_target'], 0)
        self.assertFalse(growth['size_passed'])

    def test_new_region_can_fail_with_perfect_common_pixel_jaccard(self):
        previous = WindowRecord(3, (0, 0, 20, 1), target_count=10, far_count=10,
                                valid_ids=list(range(20)), target_ids=list(range(10)))
        ids = np.arange(50)
        target = (ids < 10) | ((ids >= 20) & (ids < 35))
        far = ~target
        z = np.where(target, 1.5, 2.4); z[20:35] += .03
        current = WindowRecord(5, (0, 0, 50, 1), valid_ids=ids.tolist(), target_ids=ids[target].tolist())
        self.assertEqual(shared_membership(previous, current), 1.)
        growth = growth_evidence(previous, ids, target, far, z, Config())
        self.assertTrue(growth['size_passed'])
        self.assertFalse(growth['consistency_passed'])
        self.assertAlmostEqual(growth['mean_difference_m'], .03)

    def test_dense_radius_steps_cannot_manufacture_consecutive_passes(self):
        s = generate('clean')
        # 该用例检验两窗确认机制本身，默认值已改为单窗通过即接受，故显式固定为 2。
        cfg = Config(radii=(3, 5, 8, 9, 10, 11), consecutive=2)
        obj = measure_frame(s.frame, s.detections, s.intrinsics, cfg).objects[0]
        self.assertEqual(obj.status, 'rejected')
        for ep in obj.endpoints:
            self.assertLess(max(r.consecutive_passes for r in ep.history), cfg.consecutive)
            self.assertFalse(ep.history[-1].checks['growth'])

    def test_plane_rank_and_coverage_failures_are_traceable(self):
        cfg = Config()
        ids = np.arange(100, 130)
        fit = plane_evidence(ids, np.full(30, 1.5), 100, (15, 1), 1.5, cfg)
        self.assertFalse(fit['reliable'])
        self.assertIn('二维', fit['reason'])
        ys, xs = np.mgrid[:8, :8]
        ids = (ys * 100 + xs).ravel()
        mask = np.ones(len(ids), bool)
        coverage = coverage_evidence(ids, 100, mask, mask, mask, replace(cfg, coverage_cell_px=10))
        self.assertEqual(coverage['target'], 1)
        self.assertFalse(coverage['passed'])

    def test_config_compatibility_and_invalid_thresholds(self):
        self.assertEqual(Config.from_dict({'confidence_min': .8}).anchor_radii, (3, 5))
        self.assertEqual(Config.from_dict({'anchor_radii': [2, 4]}).anchor_radii, (2, 4))
        for values in ({'anchor_radii': [5, 3]}, {'subgroup_min_fraction': .6},
                       {'coverage_cell_px': 0}, {'growth_min_fraction': 1.1}, {'plane_min_samples': 2},
                       {'endpoint_depth_mode': 'unknown'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_diagnostics_and_export_preserve_actual_evidence(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_font(app)
        s = generate('slope_bias')
        cfg = Config(endpoint_depth_mode='strict_local')
        result = measure_frame(s.frame, s.detections, s.intrinsics, cfg)
        ep = result.objects[0].endpoints[0]
        plot = DiagnosticsPlot()
        plot.set_data(ep, s.frame.depth_m, s.detections[0], s.intrinsics, cfg)
        table = plot.criteria.table
        names = {table.item(i, 0).text(): i for i in range(table.rowCount())}
        self.assertEqual(table.item(names['均值代表性'], 3).text(), '未通过')
        self.assertEqual(table.item(names['连续通过次数'], 1).text(), str(ep.history[-1].consecutive_passes))
        with temporary_directory() as directory:
            export_results(directory, s.frame, s.detections, result)
            data = json.loads((Path(directory) / 'measurement.json').read_text(encoding='utf-8'))
            saved = data['objects'][0]['endpoints'][0]['history'][-1]
            self.assertEqual(saved['plane'], ep.history[-1].plane)
            self.assertEqual(saved['growth'], ep.history[-1].growth)
            self.assertIsNone(data['objects'][0]['distance_m'])
        plot.set_data(None, None)
        self.assertEqual(table.rowCount(), 0)
        plot.close()


if __name__ == '__main__':
    unittest.main()
