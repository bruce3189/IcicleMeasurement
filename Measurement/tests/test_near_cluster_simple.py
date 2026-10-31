import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import numpy as np
from icicle_measure.models import Config, Frame, Detection, Intrinsics
from icicle_measure.core import endpoint_estimate, Strategies, otsu_split, measure_frame


class NearFirstTests(unittest.TestCase):
    def setUp(self):
        self.z = np.full((81, 81), 2., dtype=float)
        self.z[30:51, 30:51] = 1.
        self.det = Detection('one', (0, 0, 80, 80), [(40.,40.), (40.,35.)])
        self.K = Intrinsics(100.,100.,40.,40.)

    def run_endpoint(self, **kwargs):
        cfg = Config(**kwargs)
        frame = Frame(np.zeros((81,81,3), np.uint8), self.z)
        return endpoint_estimate(frame, self.det, 0, self.K, cfg, Strategies(), lambda: False)

    def test_first_window_accepts_and_has_no_score(self):
        self.z[:] = 2.; self.z[35:46,35:41] = 1.
        ep = self.run_endpoint(radii=(5,))
        self.assertEqual(ep.status, 'accepted')
        self.assertEqual(len(ep.history), 1)
        self.assertIsNone(ep.confidence)
        self.assertIsNone(ep.history[0].center_change_m)
        self.assertEqual(set(ep.history[0].checks), {'near_samples','far_samples','separation'})

    def test_far_two_samples_are_insufficient(self):
        self.z[:] = 1.; self.z[35,35:37] = 2.
        ep = self.run_endpoint(radii=(5,))
        self.assertEqual(ep.status, 'rejected')
        self.assertEqual(ep.history[0].far_count, 2)
        self.assertEqual(ep.history[0].near_count, 119)

    def test_single_far_outlier_is_insufficient(self):
        self.z[:] = 1.; self.z[35,35] = 2.
        self.assertEqual(self.run_endpoint(radii=(5,)).status, 'rejected')

    def test_tiny_near_noise_rejected(self):
        self.z[:] = 2.; self.z[40,40] = 1.
        self.assertEqual(self.run_endpoint().status, 'rejected')

    def test_uniform_and_empty_do_not_manufacture_two_clusters(self):
        for value in (1., 0., float('nan')):
            self.z[:] = value
            self.assertEqual(self.run_endpoint().status, 'rejected')

    def test_mad_removes_within_cluster_outlier(self):
        z = np.r_[np.ones(100), 1.05, np.full(100,2.)]
        labels = otsu_split(z, Config())
        self.assertEqual(labels[100], 0)
        self.assertEqual(int((labels == 1).sum()), 100)

    def test_depth_change_does_not_gate_acceptance(self):
        self.z[:] = 2.; self.z[32:49,32:49] = 1.
        # First two windows contain only foreground, next includes background.
        ep = self.run_endpoint(radii=(3,5,12))
        self.assertEqual(ep.status, 'accepted')
        self.assertIsNone(ep.history[-1].center_change_m)
        self.assertNotIn('center', ep.history[-1].checks)

    def test_foreign_icicle_entering_later_cannot_change_accepted_depth(self):
        self.z[:] = 2.; self.z[35:46,35:41] = 1.
        original = self.run_endpoint(radii=(5,12,25))
        self.z[15:34,15:34] = .5
        changed = self.run_endpoint(radii=(5,12,25))
        self.assertEqual(changed.depth_m, original.depth_m)
        self.assertEqual(changed.history[-1].radius, 5)

    def test_actual_output_is_cleaned_mean_on_original_ray(self):
        ep = self.run_endpoint(radii=(12,18))
        self.assertEqual(ep.status,'accepted')
        self.assertEqual(ep.pixel, (40.,40.))
        self.assertEqual(ep.depth_m, float(self.z.ravel()[ep.history[-1].target_ids].mean()))
        self.assertEqual(ep.xyz_m, (0.,0.,ep.depth_m))

    def test_default_parameters_expose_only_active_fields(self):
        cfg = Config.from_dict(Config().active_dict())
        self.assertEqual(len(cfg.active_dict()),7)
        self.assertNotIn('confidence_min', cfg.active_dict())
        self.assertNotIn('bbox_padding', cfg.active_dict())
        self.assertEqual(cfg.endpoint_depth_mode,'near_cluster_simple')

    def test_cancel_and_missing_endpoint_do_not_output_length(self):
        self.det.keypoints[0] = None
        frame = Frame(np.zeros((81,81,3),np.uint8),self.z)
        obj=measure_frame(frame,[self.det],self.K).objects[0]
        self.assertEqual(obj.status,'skipped')
        self.assertIsNone(obj.distance_m)

    def test_first_eligible_matches_full_scan_and_stops_before_cap(self):
        from icicle_measure.window_analysis import analyze_endpoint, WindowAnalysisConfig
        from unittest.mock import patch
        self.z[:] = 2.; self.z[35:46,35:41] = 1.
        frame = Frame(np.zeros((81,81,3), np.uint8), self.z)
        full = analyze_endpoint(frame, self.det, 0, WindowAnalysisConfig())
        expected = next(w for w in full['windows'] if w['candidate'])
        with patch('icicle_measure.window_analysis.otsu_split', wraps=otsu_split) as split:
            ep = endpoint_estimate(frame, self.det, 0, self.K, Config(),
                                   Strategies(clusterer=split), lambda:False)
        self.assertEqual(ep.history[-1].bounds, tuple(expected['bounds']))
        self.assertEqual(ep.depth_m, expected['selected']['near_mean_m'])
        self.assertEqual(len(ep.history), expected['index']+1)
        self.assertLess(len(ep.history), len(full['windows']))
        self.assertEqual(split.call_count, expected['index']+1)

    def test_range_and_geometry_ignore_legacy_padding_and_source_range(self):
        self.z[:] = 6.; self.z[35:46,35:41] = 1.
        frame = Frame(np.zeros((81,81,3), np.uint8), self.z, max_depth_m=8.)
        ep = endpoint_estimate(frame, self.det, 0, self.K, Config(bbox_padding=0), Strategies(), lambda:False)
        self.assertEqual(ep.status, 'rejected')
        self.assertTrue(all(self.z.ravel()[r.valid_ids].max() <= 5 for r in ep.history))
        self.assertEqual(ep.history[-1].radius, 39)
        self.assertEqual(ep.history[-1].bounds, (1,1,80,80))

    def test_ui_has_no_old_hard_gates(self):
        from PyQt5 import QtWidgets
        from icicle_measure.ui.diagnostics import MeasurementCriteriaPanel
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel=MeasurementCriteriaPanel()
        panel.set_data(self.run_endpoint(radii=(12,)),Config())
        names=[panel.table.item(i,0).text() for i in range(panel.table.rowCount())]
        self.assertEqual(names,['近簇样本数','双群分离度 S','远簇比较样本','估计深度变化'])
        panel.close()

if __name__ == '__main__':
    unittest.main()
