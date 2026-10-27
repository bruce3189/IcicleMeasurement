import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace, asdict
import copy
import csv
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtTest
from icicle_measure.models import Detection, Intrinsics
from icicle_measure.pseudo_depth import (PseudoConfig, compose_scene, letterbox,
    transformed_intrinsics, to_z16, export_project, load_project)
from icicle_measure.pseudo_gui import PseudoDepthDialog
from icicle_measure.pseudo_models import _fit_mask, run_inference
from icicle_measure.io import load_bundle, save_input_bundle, read_json, write_json, export_results
from icicle_measure.core import measure_frame
from icicle_measure.gui import MainWindow, configure_font
from tests.helpers import temporary_directory
from tests.pseudo_fixtures import fixture


class PseudoDepthTests(unittest.TestCase):
    def test_letterbox_keeps_rays_and_instance_association(self):
        rgb = np.zeros((900, 400, 3), np.uint8)
        d = Detection('stable-id', (30., 200., 100., 700.), [(60., 699.), None])
        canvas, ds, valid, t = letterbox(rgb, [d])
        k = Intrinsics(700., 710., 199.5, 449.5)
        kk = transformed_intrinsics(k, t)
        self.assertEqual(canvas.shape, (720, 1280, 3))
        self.assertEqual(ds[0].id, 'stable-id')
        self.assertIsNone(ds[0].keypoints[1])
        u, v = d.keypoints[0]; uu, vv = ds[0].keypoints[0]
        self.assertAlmostEqual((u - k.cx) / k.fx, (uu - kk.cx) / kk.fx)
        self.assertAlmostEqual((v - k.cy) / k.fy, (vv - kk.cy) / kk.fy)
        self.assertFalse(valid[:, 0].any())

    def test_z16_rounding_zero_and_overflow(self):
        z = np.array([[0., np.nan, .5004, 1.2345, 3.9996]])
        raw = to_z16(z)
        self.assertEqual(raw.dtype, np.uint16)
        np.testing.assert_array_equal(raw[0, :2], [0, 0])
        self.assertLessEqual(np.max(np.abs(raw[0, 2:] * .001 - z[0, 2:])), .0005000001)
        with self.assertRaises(ValueError):
            to_z16(np.array([[66.]]))

    def test_background_is_not_flat_and_off_target_point_stays_background(self):
        s, cfg = fixture()
        r = compose_scene(s, cfg)
        self.assertGreater(np.ptp(r.background_m[s.valid & ~s.masks.any(axis=0)]), 1.)
        u, v = s.detections[0].keypoints[0]
        self.assertFalse(s.masks[0, int(v), int(u)])
        self.assertEqual(r.clean_raw[int(v), int(u)], to_z16(r.background_m)[int(v), int(u)])
        self.assertTrue(np.all(r.clean_raw[~s.valid] == 0))
        self.assertTrue(np.all(r.observed_raw[~s.valid] == 0))
        self.assertEqual(r.report['objects'][1]['status'], 'needs_mask_review')

    def test_surface_is_curved_and_length_prior_does_not_fit_output(self):
        s, cfg = fixture()
        a = compose_scene(s, cfg)
        b = compose_scene(s, replace(cfg, length_min_m=.8, length_max_m=1.,
                                    length_nominal_min_m=.82, length_nominal_max_m=.98))
        np.testing.assert_array_equal(a.clean_raw, b.clean_raw)
        row = cfg.height // 2
        x = cfg.width // 2
        self.assertLess(a.latent_m[row, x], a.latent_m[row, x - 5])
        self.assertFalse(a.report['objects'][0]['length_prior_used_to_fit'])

    def test_errors_are_repeatable_and_leave_complete_surface_unchanged(self):
        s, cfg = fixture()
        a = compose_scene(s, cfg); b = compose_scene(s, cfg)
        c = compose_scene(s, replace(cfg, seed=43))
        np.testing.assert_array_equal(a.observed_raw, b.observed_raw)
        np.testing.assert_array_equal(a.clean_raw, c.clean_raw)
        self.assertFalse(np.array_equal(a.observed_raw, c.observed_raw))
        self.assertEqual(a.report['hole_pixels'], round(a.report['target_pixels'] * .1))
        self.assertGreater(a.report['return_pixels'], 0)
        self.assertTrue(np.array_equal(s.masks, s.original_masks))
        no_errors = replace(cfg, hole_fraction=0, return_fraction=0, edge_fraction=0, noise_sigma_m_at_1m=0)
        d = compose_scene(s, no_errors)
        np.testing.assert_array_equal(d.observed_raw, d.clean_raw)

    def test_overlap_is_invalid_and_manual_changes_are_local(self):
        s, cfg = fixture()
        s.masks[1] = s.masks[0]
        r = compose_scene(s, cfg)
        self.assertTrue(np.all(r.observed_raw[s.masks[0]] == 0))
        self.assertGreater(r.report['overlap_pixels'], 0)
        s.masks[1] = False
        original = compose_scene(s, cfg)
        s.edits['objects']['ice-A'] = {'base_m': 1.1, 'slope_m': .02}
        s.background_offset[30:35, 40:45] = .1
        edited = compose_scene(s, cfg)
        self.assertTrue(np.all(edited.source_map[s.masks[0]] == 3))
        self.assertTrue(np.all(edited.source_map[30:35, 40:45] == 3))
        unchanged = s.valid & ~s.masks.any(axis=0)
        unchanged[30:35, 40:45] = False
        np.testing.assert_array_equal(original.clean_raw[unchanged], edited.clean_raw[unchanged])

    def test_empty_constant_or_invalid_parameters_fail_clearly(self):
        s, cfg = fixture()
        s.relative[:] = 1.
        with self.assertRaisesRegex(ValueError, '近乎常数'):
            compose_scene(s, cfg)
        for values in [{'depth_scale_m_per_unit': 0}, {'hole_fraction': 1.1},
                       {'noise_sigma_m_at_1m': -1}, {'far_m': .4}, {'seed': -1}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()
        with self.assertRaises(InterruptedError):
            s, cfg = fixture(); compose_scene(s, cfg, cancel=lambda: True)

    def test_model_rectangle_and_outside_candidates_are_rejected(self):
        mask, info = _fit_mask(np.ones((80, 40), bool), (0, 0, 39, 79), (80, 40))
        self.assertFalse(info['accepted']); self.assertFalse(mask.any())
        mask, info = _fit_mask(np.ones((80, 40), bool), (15, 20, 25, 60), (80, 40))
        self.assertFalse(info['accepted'])

    def test_export_reload_preserves_depth_edits_and_honest_geometry(self):
        s, cfg = fixture()
        s.edits['objects']['ice-A'] = {'base_m': 1.05, 'slope_m': -.01}
        r = compose_scene(s, cfg)
        with temporary_directory() as directory:
            manifest = export_project(directory, r)
            frame, ds, k = load_bundle(manifest)
            self.assertEqual(ds[0].id, 'ice-A')
            self.assertFalse(frame.metadata['undistorted'])
            self.assertTrue(frame.metadata['is_synthetic'])
            self.assertEqual(frame.metadata['geometry_mode'], 'assumed_pinhole')
            np.testing.assert_allclose(frame.depth_m, r.frame.depth_m, atol=1e-6)
            loaded = load_project(directory)
            np.testing.assert_array_equal(loaded.clean_raw, r.clean_raw)
            np.testing.assert_array_equal(loaded.observed_raw, r.observed_raw)
            self.assertEqual(loaded.scene.edits, s.edits)
            raw = cv2.imread(str(Path(directory) / 'depth_observed_z16.png'), cv2.IMREAD_UNCHANGED)
            self.assertEqual(raw.dtype, np.uint16); self.assertEqual(raw.ndim, 2)
            result = measure_frame(frame, ds, k)
            self.assertIn('原照片畸变未经验证', result.assumptions)
            export_results(Path(directory) / 'measurement', frame, ds, result)
            saved = read_json(Path(directory) / 'measurement' / 'measurement.json')
            self.assertEqual(saved['source']['input_kind'], 'pseudo_rgbd')
            with (Path(directory) / 'measurement' / 'measurement.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(row['measurement_label'] == '基于伪深度的估计距离' for row in rows))
            copied = save_input_bundle(Path(directory) / 'resaved', frame, ds, k)
            self.assertFalse(load_bundle(copied)[0].metadata['undistorted'])
            bad = read_json(manifest); bad['source']['is_synthetic'] = False
            write_json(Path(directory) / 'bad.json', bad)
            with self.assertRaisesRegex(ValueError, '去畸变'):
                load_bundle(Path(directory) / 'bad.json')


class PseudoGuiTests(unittest.TestCase):
    def test_polygon_plane_undo_and_excluded_target_restore(self):
        s, cfg = fixture(1280, 720)
        d = PseudoDepthDialog(); d.receive({'result': compose_scene(s, cfg)})
        original = d.result.clean_raw.copy()
        for x,y in [(20,20),(200,20),(200,180),(20,180)]:
            d.add_prompt({'mode':'plane','x':x,'y':y})
        self.assertEqual(len(d.plane_points),4)
        d.commit_plane(); self.wait(d)
        self.assertEqual(len(d.result.scene.edits['background_planes']),1)
        self.assertEqual(d.plane_points,[])
        self.assertFalse(np.array_equal(original,d.result.clean_raw))
        d.undo_plane(); self.wait(d)
        np.testing.assert_array_equal(original,d.result.clean_raw)
        d.exclude_object()
        self.assertFalse(d.result.scene.masks[0].any())
        self.assertEqual(len(d.result.scene.detections),2)
        d.reset_mask()
        np.testing.assert_array_equal(d.result.scene.masks[0],s.original_masks[0])
        scene = d.result.scene
        scene.masks[1] = scene.masks[0].copy()
        keep = scene.masks[0].copy()
        d.assign_overlap()
        np.testing.assert_array_equal(scene.masks[0],keep)
        self.assertFalse(scene.masks[1].any())
        self.assertEqual(scene.edits['actions'][-1]['affected_ids'],['ice-B'])
        d.close()

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_font(cls.app)

    def wait(self, dialog):
        deadline = time.monotonic() + 12
        while dialog.thread and time.monotonic() < deadline:
            self.app.processEvents(); QtTest.QTest.qWait(10)
        self.assertIsNone(dialog.thread)

    def test_brush_recompute_save_reopen_and_send_with_source_label(self):
        s, cfg = fixture(1280, 720)
        r = compose_scene(s, cfg)
        d = PseudoDepthDialog(); d.receive({'result': r, 'root': None}); d.show()
        self.app.processEvents()
        self.assertFalse(d.dirty)
        before = s.masks.copy()
        x, y = 640, 300
        d.paint_mask({'start': (x, y), 'end': (x, y + 10), 'mode': 'erase'})
        self.assertFalse(np.array_equal(before, s.masks))
        self.assertTrue(d.dirty)
        d.auto_base.setChecked(False); d.base.setValue(1.15)
        d.apply_edits(); self.wait(d)
        self.assertFalse(d.dirty)
        self.assertEqual(d.result.scene.edits['objects']['ice-A']['base_m'], 1.15)
        self.assertTrue(np.all(d.result.frame.depth_m[~s.valid] == 0))
        with temporary_directory() as folder:
            with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=folder):
                d.save_project(); self.wait(d)
            reopened = load_project(folder)
            np.testing.assert_array_equal(reopened.observed_raw, d.result.observed_raw)
        window = MainWindow()
        d.measurement_requested.connect(window.receive_pseudo_input)
        d.send_measurement(); self.wait(window)
        self.assertTrue(window.frame.metadata['is_synthetic'])
        self.assertIn('伪深度', window.input_note.text())
        self.assertIn('伪深度', window.table.horizontalHeaderItem(6).text())
        window.close(); d.close(); self.app.processEvents()

    def test_background_edit_parameter_dirty_and_worker_cancel(self):
        s, cfg = fixture(1280, 720)
        d = PseudoDepthDialog(); d.receive({'result': compose_scene(s, cfg)})
        d.paint_mask({'start': (50, 50), 'end': (100, 50), 'mode': 'background'})
        self.assertTrue(np.any(s.background_offset != 0))
        d.apply_edits(); self.wait(d)
        d.settings['hole_fraction'].setValue(.2)
        self.assertTrue(d.dirty)
        def slow(progress, cancel):
            while not cancel():
                time.sleep(.01)
            raise InterruptedError('cancelled fixture')
        d.launch(slow, 'cancel test'); d.cancel(); self.wait(d)
        self.assertTrue(d.generate_button.isEnabled())
        self.assertIn('cancelled', d.log.toPlainText())
        d.close()

    def test_object_edits_survive_switching_before_apply(self):
        s, cfg = fixture(1280, 720)
        d = PseudoDepthDialog(); d.receive({'result': compose_scene(s, cfg)})
        d.auto_base.setChecked(False); d.base.setValue(1.35)
        d.object_choice.setCurrentIndex(1)
        d.object_choice.setCurrentIndex(0)
        self.assertFalse(d.auto_base.isChecked())
        self.assertAlmostEqual(d.base.value(), 1.35)
        self.assertTrue(d.dirty)
        d.apply_edits(); self.wait(d)
        self.assertAlmostEqual(d.result.report['objects'][0]['base_m'], 1.35)
        d.close()

    def test_generator_fits_small_screen_with_scrollable_parameters(self):
        s, cfg = fixture(1280, 720)
        window = MainWindow()
        d = PseudoDepthDialog(window)
        d.receive({'result': compose_scene(s, cfg)})
        d.resize(1366, 768); d.show()
        d.view_kind.setCurrentIndex(d.view_kind.findData('observed'))
        self.app.processEvents()
        self.assertLessEqual(d.height(), 768)
        self.assertTrue(d.colorbar.isVisible())
        self.assertLess(d.status.geometry().bottom(), d.height())
        self.assertTrue(d.measure_button.isEnabled())
        self.assertIsInstance(d.panel.widget(1), QtWidgets.QScrollArea)
        d.close(); window.close()


if __name__ == '__main__':
    unittest.main()
