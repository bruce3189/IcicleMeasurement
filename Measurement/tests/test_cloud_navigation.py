import unittest
import numpy as np
from PyQt5 import QtWidgets,QtCore,QtGui,QtTest
from icicle_measure.gui import DepthTerrainCanvas,MainWindow
from icicle_measure.simulator import generate
from icicle_measure.core import measure_frame


class CloudNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def canvas(self):
        s=generate('edge',42)
        measured=measure_frame(s.frame,s.detections,s.intrinsics)
        c=DepthTerrainCanvas(); c.resize(900,650)
        c.set_data(measured.objects[0].endpoints[0],s.frame.depth_m,s.detections[0],s.intrinsics)
        c.show(); self.app.processEvents(); c.grab()
        self.addCleanup(c.close)
        return c

    def project(self,c,index):
        c.grab(); plot=c.plot_rect(); scale=c._fit_cache[1]*c.zoom
        center=np.asarray([plot.center().x(),plot.center().y()])-c._fit_cache[2]*scale+[c.pan.x(),c.pan.y()]
        return c._project(c.display_vertices()[[index]],scale,*center)[0][0]

    def select_visible(self,c):
        c.grab(); i=int(c._proj_index[len(c._proj_index)//2])
        c.selected_pixel_id=int(c.sample_pixel_ids[i]); return i

    def test_full_zoom_range_keeps_cursor_anchor_at_both_limits(self):
        c=self.canvas(); index=self.select_visible(c)
        anchor=self.project(c,index); pos=QtCore.QPointF(*anchor)
        xyz=c.relative_m.copy(); rotation=c.rotation.copy()
        for factor in (80,2,2,.001,.1,2,10,10):
            c.zoom_at(pos,factor)
            self.assertGreaterEqual(c.zoom,.05); self.assertLessEqual(c.zoom,80.)
            np.testing.assert_allclose(self.project(c,index),anchor,atol=1e-7)
        np.testing.assert_array_equal(c.relative_m,xyz)
        np.testing.assert_array_equal(c.rotation,rotation)

    def test_shift_wheel_is_fine_grained_and_horizontal_wheel_is_ignored(self):
        c=self.canvas(); pos=QtCore.QPointF(c.plot_rect().center())
        def wheel(modifier,angle=QtCore.QPoint(0,120)):
            e=QtGui.QWheelEvent(pos,pos,QtCore.QPoint(),angle,QtCore.Qt.NoButton,modifier,QtCore.Qt.NoScrollPhase,False)
            c.wheelEvent(e)
            while c._zoom_animation.state()==QtCore.QAbstractAnimation.Running:
                self.app.processEvents();QtTest.QTest.qWait(5)
        wheel(QtCore.Qt.ShiftModifier)
        self.assertAlmostEqual(c.zoom,c.WHEEL_STEP**.2)
        c.reset_view(); wheel(QtCore.Qt.NoModifier)
        self.assertAlmostEqual(c.zoom,c.WHEEL_STEP)
        old=c.zoom; wheel(QtCore.Qt.NoModifier,QtCore.QPoint(120,0))
        self.assertEqual(c.zoom,old)

    def test_focused_sample_stays_at_centre_through_full_rotation_and_resize(self):
        c=self.canvas(); i=self.select_visible(c)
        xyz=c.relative_m.copy(); origin=c.origin_xyz_m.copy()
        self.assertTrue(c.focus_selected())
        np.testing.assert_array_equal(c.orbit_pivot_relative,xyz[i])
        for axis,angle in [('x',180),('y',180),('z',180),('x',-180),('y',-180)]:
            c.rotate_axis(axis,angle)
            np.testing.assert_allclose(self.project(c,i),[c.plot_rect().center().x(),c.plot_rect().center().y()],atol=1e-7)
        c.resize(1250,850); self.app.processEvents()
        np.testing.assert_allclose(self.project(c,i),[c.plot_rect().center().x(),c.plot_rect().center().y()],atol=1e-7)
        np.testing.assert_array_equal(c.relative_m,xyz)
        np.testing.assert_array_equal(c.origin_xyz_m,origin)

    def test_endpoint_return_and_reset_do_not_move_geometry(self):
        c=self.canvas(); i=self.select_visible(c); before=c.vertices.copy()
        c.focus_selected(); c.zoom_at(QtCore.QPointF(c.plot_rect().center()),3)
        zoom=c.zoom; c.return_to_endpoint(); c.grab()
        self.assertEqual(c.zoom,zoom); self.assertIsNone(c.orbit_focus_id)
        plot=c.plot_rect(); scale=c._fit_cache[1]*c.zoom
        center=np.asarray([plot.center().x(),plot.center().y()])-c._fit_cache[2]*scale+[c.pan.x(),c.pan.y()]
        screen,_=c._project(np.zeros((1,3)),scale,*center)
        np.testing.assert_allclose(screen[0],[plot.center().x(),plot.center().y()],atol=1e-7)
        c.reset_view(); self.assertEqual(c.zoom,1.)
        np.testing.assert_array_equal(c.vertices,before)

    def test_fit_uses_cloud_not_origin_guides_and_empty_focus_is_safe(self):
        c=self.canvas()
        c.category_visibility[:]=False; c.category_visibility[3]=True
        c.fit_view(); c.grab()
        selected=c._proj_screen
        self.assertGreater(len(selected),0)
        fraction=np.ptp(selected,axis=0)/[c.plot_rect().width(),c.plot_rect().height()]
        self.assertGreater(float(fraction.max()),.8)
        empty=DepthTerrainCanvas(); self.assertFalse(empty.focus_selected())
        empty.return_to_endpoint(); empty.close()

    def test_inline_expansion_keeps_same_canvas_state_and_escape_restores_layout(self):
        w=MainWindow(); self.addCleanup(w.close)
        s=generate('clean'); w.set_input(s.frame,s.detections,s.intrinsics)
        w.receive_result(measure_frame(s.frame,s.detections,s.intrinsics))
        w.resize(1366,768); w.show(); self.app.processEvents()
        c=w.plot.terrain.canvas; self.select_visible(c); c.focus_selected()
        rotation=c.rotation.copy(); xyz=c.relative_m.copy(); zoom=c.zoom
        w.plot.terrain.expand_button.setChecked(True); self.app.processEvents(); c.grab()
        self.assertTrue(w._cloud_expanded)
        self.assertFalse(w.table.isVisible()); self.assertFalse(w.image.isVisible())
        self.assertGreater(c.width(),1000); self.assertGreater(c.height(),450)
        self.assertIs(c,w.plot.terrain.canvas)
        np.testing.assert_array_equal(c.rotation,rotation)
        self.assertEqual(c.zoom,zoom)
        QtTest.QTest.keyClick(c,QtCore.Qt.Key_Escape); self.app.processEvents()
        self.assertFalse(w._cloud_expanded)
        self.assertTrue(w.table.isVisible()); self.assertTrue(w.image.isVisible())
        np.testing.assert_array_equal(c.relative_m,xyz)
