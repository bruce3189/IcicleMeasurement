import unittest
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from tests.pseudo_fixtures import fixture
from icicle_measure.pseudo_depth import compose_scene, _map_relative
from icicle_measure.pixel_inspector import inspect_pixel, pixel_text
from icicle_measure.pseudo_gui import EditView, PseudoDepthDialog


class PixelInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_observed_depth_and_xyz_are_actual_pixel_values(self):
        scene,cfg=fixture(); r=compose_scene(scene,cfg)
        v,u=np.argwhere((r.instance_map>0)&(r.observed_raw>0))[0]
        info=inspect_pixel(r,int(u),int(v))
        z=float(r.frame.depth_m[v,u]); k=scene.intrinsics
        self.assertEqual(info['depth_m'],z)
        self.assertEqual(info['current_raw'],int(r.observed_raw[v,u]))
        np.testing.assert_allclose(info['xyz_m'],[(u-k.cx)*z/k.fx,(v-k.cy)*z/k.fy,z])
        self.assertEqual(info['instance'],'ice-A')

    def test_hole_does_not_borrow_complete_depth_or_create_xyz(self):
        scene,cfg=fixture(); r=compose_scene(scene,cfg)
        v,u=np.argwhere((r.uncertainty&16)>0)[0]
        observed=inspect_pixel(r,int(u),int(v),'observed')
        complete=inspect_pixel(r,int(u),int(v),'complete')
        self.assertFalse(observed['valid']); self.assertIsNone(observed['xyz_m'])
        self.assertIn('观测空洞',observed['source'])
        self.assertTrue(complete['valid']); self.assertGreater(complete['depth_m'],0)

    def test_relative_layer_and_padding_are_not_metric_points(self):
        scene,cfg=fixture(); r=compose_scene(scene,cfg)
        info=inspect_pixel(r,100,100,'relative')
        self.assertIsNone(info['depth_m']); self.assertIsNone(info['xyz_m'])
        compact,detail=pixel_text(info)
        self.assertIn('无米制单位',compact)
        self.assertIn('非米',detail)
        padding=inspect_pixel(r,0,0)
        self.assertIsNone(padding['original_pixel']); self.assertFalse(padding['valid'])
        self.assertIsNone(inspect_pixel(r,320,20))

    def test_model_and_background_are_not_read_from_observed_layer(self):
        scene,cfg=fixture(); r=compose_scene(scene,cfg)
        mapped,_=_map_relative(scene.relative,scene.valid,cfg)
        v,u=np.argwhere(r.instance_map>0)[0]
        for layer,array in [('model',mapped),('background',r.background_m)]:
            info=inspect_pixel(r,int(u),int(v),layer,model_depth=mapped)
            self.assertEqual(info['depth_m'],float(array[v,u]))
            self.assertIsNone(info['current_raw'])

    def test_view_mapping_survives_zoom_and_last_pixel_is_in_bounds(self):
        view=EditView(); view.resize(700,500); view.show()
        view.set_image(np.zeros((240,320,3),np.uint8),fit=True)
        self.app.processEvents()
        received=[]; view.pixel_hovered.connect(received.append)
        for scale in (1.,2.5):
            view.resetTransform(); view.scale(scale,scale)
            self.app.processEvents()
            pos=view.mapFromScene(QtCore.QPointF(150.25,100.25))
            expected=view.pixel_at(pos)
            move=QtGui.QMouseEvent(QtCore.QEvent.MouseMove,QtCore.QPointF(pos),QtCore.Qt.NoButton,QtCore.Qt.NoButton,QtCore.Qt.NoModifier)
            self.app.sendEvent(view.viewport(),move); self.app.processEvents()
            self.assertEqual(received[-1],expected)
            self.assertEqual(expected,(150,100))
        view.resetTransform(); view.scale(10.,10.)
        pos=view.mapFromScene(QtCore.QPointF(319.9,239.9))
        self.assertEqual(view.pixel_at(pos),(319,239))
        view.close()

    def test_dialog_hover_updates_layer_and_clears_when_leaving(self):
        s,cfg=fixture(1280,720); r=compose_scene(s,cfg)
        dialog=PseudoDepthDialog(); dialog.receive({'result':r}); dialog.show()
        self.app.processEvents()
        dialog.view_kind.setCurrentIndex(dialog.view_kind.findData('observed'))
        dialog.show_pixel_info((640,300))
        self.assertIn('像素 (640, 300)',dialog.hover_readout.text())
        self.assertIn('当前图层：observed',dialog.pixel_details.toPlainText())
        dialog.view_kind.setCurrentIndex(dialog.view_kind.findData('relative'))
        dialog.show_pixel_info((640,300))
        self.assertIn('无米制单位',dialog.hover_readout.text())
        dialog.dirty=True; dialog.show_pixel_info((640,300))
        self.assertIn('尚未应用',dialog.hover_readout.text())
        dialog.show_pixel_info(None)
        self.assertNotIn('(640, 300)',dialog.hover_readout.text())
        dialog.close()
