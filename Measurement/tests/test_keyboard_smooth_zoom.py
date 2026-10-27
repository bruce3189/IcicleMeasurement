import unittest
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets,QtTest
from icicle_measure.gui import DepthTerrainCanvas
from icicle_measure.simulator import generate
from icicle_measure.core import measure_frame


class KeyboardSmoothZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def canvas(self):
        s=generate('clean',42)
        c=DepthTerrainCanvas();c.resize(900,650)
        result=measure_frame(s.frame,s.detections,s.intrinsics)
        c.set_data(result.objects[0].endpoints[0],s.frame.depth_m,s.detections[0],s.intrinsics)
        c.show();self.app.processEvents();c.grab()
        self.addCleanup(c.close);return c

    def screen(self,c,indices):
        c.grab();p=c.plot_rect();scale=c._fit_cache[1]*c.zoom
        center=np.array([p.center().x(),p.center().y()])-c._fit_cache[2]*scale+[c.pan.x(),c.pan.y()]
        return c._project(c.display_vertices()[indices],scale,*center)[0]

    def test_wasd_directions_and_continuous_motion_stop_on_release(self):
        c=self.canvas();geometry=c.relative_m.copy();rotation=c.rotation.copy()
        for key,sign,axis in [(QtCore.Qt.Key_W,-1,'y'),(QtCore.Qt.Key_S,1,'y'),
                              (QtCore.Qt.Key_A,-1,'x'),(QtCore.Qt.Key_D,1,'x')]:
            c.pan=QtCore.QPointF()
            QtTest.QTest.keyPress(c,key);QtTest.QTest.qWait(80);QtTest.QTest.keyRelease(c,key)
            self.assertGreater(sign*getattr(c.pan,axis)(),6.)
            self.assertFalse(c._pan_timer.isActive())
            stopped=QtCore.QPointF(c.pan);QtTest.QTest.qWait(35)
            self.assertEqual(c.pan,stopped)
        np.testing.assert_array_equal(c.relative_m,geometry)
        np.testing.assert_array_equal(c.rotation,rotation)

    def test_shift_is_fine_pan_and_focus_loss_stops_held_keys(self):
        c=self.canvas()
        QtTest.QTest.keyPress(c,QtCore.Qt.Key_D);QtTest.QTest.keyRelease(c,QtCore.Qt.Key_D)
        normal=c.pan.x();c.pan=QtCore.QPointF()
        QtTest.QTest.keyPress(c,QtCore.Qt.Key_D,QtCore.Qt.ShiftModifier)
        QtTest.QTest.keyRelease(c,QtCore.Qt.Key_D,QtCore.Qt.ShiftModifier)
        self.assertAlmostEqual(c.pan.x()/normal,.2)
        QtTest.QTest.keyPress(c,QtCore.Qt.Key_W)
        self.app.sendEvent(c,QtGui.QFocusEvent(QtCore.QEvent.FocusOut))
        before=QtCore.QPointF(c.pan);QtTest.QTest.qWait(40)
        self.assertEqual(c.pan,before);self.assertFalse(c._pan_keys)
        self.assertFalse(c._pan_timer.isActive())

    def test_text_editing_does_not_move_cloud(self):
        c=self.canvas();line=QtWidgets.QLineEdit();self.addCleanup(line.close)
        line.show();line.setFocus();self.app.processEvents()
        old=QtCore.QPointF(c.pan)
        QtTest.QTest.keyClicks(line,'wasd')
        self.assertEqual(line.text(),'wasd');self.assertEqual(c.pan,old)

    def test_smooth_zoom_keeps_anchor_in_every_intermediate_frame(self):
        c=self.canvas();i=int(c._proj_index[len(c._proj_index)//2]);j=(i+1)%len(c.vertices)
        before=c.relative_m.copy();initial=self.screen(c,[i,j]);anchor=QtCore.QPointF(*initial[0])
        c.animate_zoom_at(anchor,1.8)
        last=1.
        for milliseconds in (10,30,55,80,110):
            c._zoom_animation.setCurrentTime(milliseconds)
            self.assertGreaterEqual(c.zoom,last);last=c.zoom
            after=self.screen(c,[i,j])
            np.testing.assert_allclose(after[0],initial[0],atol=1e-7)
            self.assertAlmostEqual(np.linalg.norm(after[1]-after[0])/np.linalg.norm(initial[1]-initial[0]),c.zoom,places=8)
        self.assertAlmostEqual(c.zoom,1.8)
        np.testing.assert_array_equal(c.relative_m,before)

    def test_rapid_wheel_inputs_accumulate_and_pan_cancels_animation(self):
        c=self.canvas();pos=QtCore.QPointF(c.plot_rect().center())
        for _ in range(8):c.animate_zoom_at(pos,c.WHEEL_STEP)
        self.assertAlmostEqual(c._zoom_target,1.08**8)
        c._zoom_animation.setCurrentTime(110)
        self.assertAlmostEqual(c.zoom,1.08**8)
        c.animate_zoom_at(pos,1.5)
        c._zoom_animation.setCurrentTime(40)
        QtTest.QTest.keyPress(c,QtCore.Qt.Key_A);QtTest.QTest.keyRelease(c,QtCore.Qt.Key_A)
        zoom=c.zoom
        self.assertEqual(c._zoom_animation.state(),QtCore.QAbstractAnimation.Stopped)
        QtTest.QTest.qWait(130);self.assertEqual(c.zoom,zoom)

    def test_reversing_wheel_does_not_continue_towards_old_target(self):
        c=self.canvas();pos=QtCore.QPointF(c.plot_rect().center())
        c.animate_zoom_at(pos,2.);c._zoom_animation.setCurrentTime(30)
        current=c.zoom;c.animate_zoom_at(pos,.8)
        self.assertLess(c._zoom_target,current)
        c._zoom_animation.setCurrentTime(110)
        self.assertAlmostEqual(c.zoom,current*.8)

    def test_uniform_xyz_and_equal_depth_xy_proportions(self):
        c=self.canvas()
        np.testing.assert_array_equal(c.vertices,c.relative_m*c.cloud_scale)
        c.rotation=np.array([1.,0.,0.,0.]);c.orbit_pivot_relative=np.zeros(3)
        cube=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        screen,_=c._project(cube,10.,0.,0.)
        self.assertAlmostEqual(np.linalg.norm(screen[1]-screen[0])/np.linalg.norm(screen[2]-screen[0]),1.)

    def test_hiding_canvas_cancels_pan_and_animation(self):
        c=self.canvas();c.animate_zoom_at(QtCore.QPointF(c.plot_rect().center()),2.)
        QtTest.QTest.keyPress(c,QtCore.Qt.Key_W)
        c.hide();self.app.processEvents()
        self.assertFalse(c._pan_timer.isActive())
        self.assertFalse(c._pan_keys)
        self.assertEqual(c._zoom_animation.state(),QtCore.QAbstractAnimation.Stopped)
