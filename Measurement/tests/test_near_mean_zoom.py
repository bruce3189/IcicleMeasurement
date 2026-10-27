import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from dataclasses import replace
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from icicle_measure.models import Config, Detection
from icicle_measure.simulator import generate
from icicle_measure.core import measure_frame
from icicle_measure.gui import DepthTerrainCanvas, DiagnosticsPlot, MainWindow, configure_font


class NearMeanZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_font(cls.app)

    def test_off_target_pixels_recover_near_mean_without_moving_rays(self):
        s = generate('noisy', 42)
        det = Detection('one', (140., 40., 180., 200.), [(170., 200.), (170., 40.)])
        obj = measure_frame(s.frame, [det], s.intrinsics).objects[0]
        self.assertEqual(obj.status, 'accepted')
        for i, ep in enumerate(obj.endpoints):
            u, v = ep.pixel
            self.assertEqual(ep.pixel, det.keypoints[i])
            self.assertGreater(s.frame.depth_m[int(v), int(u)], 2.)
            self.assertAlmostEqual(ep.depth_m, s.frame.depth_m.ravel()[ep.history[-1].target_ids].mean())
            self.assertLess(abs(ep.depth_m - [1.5, 1.3][i]), .010)
            self.assertAlmostEqual(ep.xyz_m[0], (u - s.intrinsics.cx) * ep.depth_m / s.intrinsics.fx)
            self.assertTrue(all(2 * r.radius + 1 <= 40 for r in ep.history))
            self.assertGreater(ep.history[-1].anchor['radius_px'], 5)
        strict = measure_frame(s.frame, [det], s.intrinsics, Config(endpoint_depth_mode='strict_local'))
        self.assertEqual(strict.objects[0].status, 'rejected')

    def test_slope_advisories_are_visible_and_do_not_replace_output(self):
        s = generate('slope_bias')
        result = measure_frame(s.frame, s.detections, s.intrinsics)
        obj = result.objects[0]
        self.assertEqual(obj.status, 'accepted')
        plot = DiagnosticsPlot()
        ep = obj.endpoints[0]
        record = ep.history[-1]
        self.assertNotIn('representative', record.checks)
        self.assertFalse(record.advisories['representative'])
        self.assertFalse(record.advisories['ring'])
        self.assertTrue(record.warnings)
        self.assertAlmostEqual(ep.depth_m, s.frame.depth_m.ravel()[record.target_ids].mean())
        self.assertGreater(abs(ep.depth_m - s.truth[obj.id]['depth_m'][0]), .010)
        plot.set_data(ep, s.frame.depth_m, s.detections[0], s.intrinsics, Config())
        table = plot.criteria.table
        rows = {table.item(i, 0).text(): i for i in range(table.rowCount())}
        self.assertEqual(table.item(rows['均值代表性'], 3).text(), '提示（不阻断）')
        self.assertIn('近簇均值近似', plot.criteria.summary.text())
        self.assertIn('仅提示', plot.criteria.summary.text())
        plot.close()

    def test_mode_switch_invalidates_previous_results_and_syncs_config(self):
        window = MainWindow()
        s = generate('clean')
        window.set_input(s.frame, s.detections, s.intrinsics, 'test')
        window.result = measure_frame(s.frame, s.detections, s.intrinsics)
        window.depth_mode.setCurrentIndex(1)
        self.assertEqual(window.config.endpoint_depth_mode, 'strict_local')
        self.assertIsNone(window.result)
        self.assertEqual(window.plot.criteria.table.rowCount(), 0)
        window.set_config(replace(window.config, endpoint_depth_mode='near_cluster_mean'))
        self.assertEqual(window.depth_mode.currentData(), 'near_cluster_mean')
        window.set_busy(True)
        self.assertFalse(window.depth_mode.isEnabled())
        window.set_busy(False)
        window.close()

    def test_wheel_keeps_cursor_anchor_and_geometry_after_rotation(self):
        s = generate('edge', 42)
        ep = measure_frame(s.frame, s.detections, s.intrinsics).objects[0].endpoints[0]
        canvas = DepthTerrainCanvas()
        canvas.resize(900, 650)
        canvas.set_data(ep, s.frame.depth_m, s.detections[0], s.intrinsics)
        canvas.rotate_axis('y', 22)
        canvas.pan = QtCore.QPointF(17., -12.)
        canvas.show()
        canvas.grab()
        self.assertGreater(len(canvas._proj_index), 0)
        index = int(canvas._proj_index[len(canvas._proj_index) // 2])
        def screen():
            canvas.grab()
            p = canvas.rect().adjusted(12, 42, -12, -44)
            scale = canvas._fit_cache[1] * canvas.zoom
            center = np.array([p.center().x(), p.center().y()]) - canvas._fit_cache[2] * scale
            center += [canvas.pan.x(), canvas.pan.y()]
            return canvas._project(canvas.display_vertices()[[index]], scale, *center)[0][0]
        anchor = screen().copy()
        pos = QtCore.QPointF(*anchor)
        before, rotation = canvas.relative_m.copy(), canvas.rotation.copy()
        def wheel(angle=(0, 120), pixel=(0, 0)):
            event = QtGui.QWheelEvent(pos, pos, QtCore.QPoint(*pixel), QtCore.QPoint(*angle),
                                     QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
            canvas.wheelEvent(event)
            while canvas._zoom_animation.state()==QtCore.QAbstractAnimation.Running:
                self.app.processEvents();QtTest.QTest.qWait(5)
        wheel()
        self.assertAlmostEqual(canvas.zoom, canvas.WHEEL_STEP)
        np.testing.assert_allclose(screen(), anchor, atol=1e-8)
        wheel((0, -120))
        self.assertAlmostEqual(canvas.zoom, 1.)
        np.testing.assert_allclose(screen(), anchor, atol=1e-8)
        state = canvas.zoom, canvas.pan.x(), canvas.pan.y()
        wheel((120, 0)); wheel((0, 0))
        self.assertEqual(state, (canvas.zoom, canvas.pan.x(), canvas.pan.y()))
        wheel((0, 0), (0, 2))
        self.assertGreater(canvas.zoom, 1.)
        self.assertLess(canvas.zoom, 1.01)
        np.testing.assert_allclose(screen(), anchor, atol=1e-8)
        for factor in [100., 2., .0001, .5]:
            canvas.zoom_at(pos, factor)
            np.testing.assert_allclose(screen(), anchor, atol=1e-8)
        np.testing.assert_array_equal(canvas.relative_m, before)
        np.testing.assert_array_equal(canvas.rotation, rotation)
        canvas.close()


if __name__ == '__main__':
    unittest.main()
