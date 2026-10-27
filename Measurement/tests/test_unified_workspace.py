import copy
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtTest
from tests.pseudo_fixtures import fixture
from tests.helpers import temporary_directory
from icicle_measure.pseudo_depth import compose_scene
from icicle_measure.pseudo_gui import PseudoDepthPanel
from icicle_measure.gui import MainWindow
from icicle_measure.core import measure_frame
from icicle_measure.render import overlay, distance_summary, verified_distance, draw_selected_distance
from icicle_measure.simulator import generate


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def pump(self, predicate, timeout=12):
        end=time.monotonic()+timeout
        while not predicate() and time.monotonic()<end:
            self.app.processEvents(); QtTest.QTest.qWait(5)
        self.assertTrue(predicate(),'GUI task did not settle')

    def window(self):
        w=MainWindow()
        with patch.object(PseudoDepthPanel,'choose_sample'):
            w.open_pseudo_depth()
        self.addCleanup(self.shutdown,w)
        return w,w.pseudo_dialog

    def shutdown(self,w):
        w.close()
        self.pump(lambda:w.thread is None and (w.pseudo_dialog is None or w.pseudo_dialog.thread is None))
        self.app.processEvents()

    def wait(self,w):
        self.pump(lambda:w.thread is None and w._pending_pseudo is None and w.pseudo_dialog.thread is None)

    def generated(self,base=1.25,tag=1):
        s,cfg=fixture(1280,720)
        s.edits['objects']['ice-A']={'base_m':base,'slope_m':0}
        r=compose_scene(s,cfg)
        r.frame.metadata['test_tag']=tag
        r.clean_frame.metadata['test_tag']=tag
        return r

    def test_embedded_tabs_automeasure_selection_and_view_preservation(self):
        w,p=self.window(); w.resize(1366,768); w.show()
        self.assertFalse(p.isWindow())
        self.assertEqual([w.workspace.tabText(i) for i in range(2)],['场景生成','测距分析'])
        p.receive({'result':self.generated()}); self.wait(w)
        self.assertLessEqual(w.height(),768)
        self.assertIs(p.measurement,w.result)
        self.assertIn('观测深度',p.distance_note.text())
        p.object_choice.setCurrentIndex(1)
        self.assertEqual(w.selected()[0].id,'ice-B')
        w.table.selectRow(1)
        self.assertEqual(p.object_choice.currentData(),'ice-A')
        self.assertEqual(w.selected()[1],1)
        p.view.scale(1.4,1.4); transform=p.view.transform()
        w.plot.terrain.canvas.rotate_axis('y',15)
        rotation=w.plot.terrain.canvas.rotation.copy()
        for index in (1,0,1,0):
            w.workspace.setCurrentIndex(index); self.app.processEvents()
        self.assertEqual(p.view.transform(),transform)
        np.testing.assert_array_equal(w.plot.terrain.canvas.rotation,rotation)
        self.assertEqual(w.selected()[1],1)

    def test_layer_switch_edit_and_config_change_invalidate_and_recompute(self):
        w,p=self.window(); p.receive({'result':self.generated()}); self.wait(w)
        p.view_kind.setCurrentIndex(p.view_kind.findData('complete')); self.wait(w)
        self.assertEqual(w.frame.metadata['depth_variant'],'complete')
        self.assertIn('完整表面',w.distance_note.text())
        p.base.setValue(1.4) # Auto flag must be disabled to edit the baseline.
        p.auto_base.setChecked(False)
        self.assertIsNone(w.result); self.assertIsNone(p.measurement)
        self.assertFalse(w.export_button.isEnabled())
        self.assertIn('未应用',p.distance_note.text())
        p.apply_edits(); self.wait(w)
        self.assertIsNotNone(w.result)
        self.assertEqual(w.frame.metadata['depth_variant'],'complete')
        before=w.result
        w.set_config(replace(w.config,confidence_min=.82)); self.wait(w)
        self.assertIsNot(before,w.result)
        self.assertEqual(w.result.config['confidence_min'],.82)

    def test_only_latest_pending_input_is_measured_and_old_completion_is_ignored(self):
        w,p=self.window()
        entered=threading.Event(); release=threading.Event(); calls=[]
        def slow(frame,ds,k,cfg,**kwargs):
            calls.append(frame.metadata['test_tag'])
            if len(calls)==1:
                entered.set(); release.wait(5)
            # Deliberately ignore cancellation to test result revision guards.
            return measure_frame(frame,ds,k,cfg)
        with patch('icicle_measure.gui.measure_frame',side_effect=slow):
            p.receive({'result':self.generated(tag=1)})
            self.pump(entered.is_set)
            p.receive({'result':self.generated(1.35,tag=2)})
            p.receive({'result':self.generated(1.45,tag=3)})
            self.assertIsNone(w.result)
            release.set(); self.wait(w)
        self.assertEqual(calls,[1,3])
        self.assertEqual(w.frame.metadata['test_tag'],3)
        self.assertEqual(w.result.source['test_tag'],3)
        self.assertIs(p.measurement,w.result)

    def test_edit_during_running_measurement_suppresses_late_result(self):
        w,p=self.window(); entered=threading.Event(); release=threading.Event()
        def slow(*args,**kwargs):
            entered.set(); release.wait(5)
            return measure_frame(*args)
        with patch('icicle_measure.gui.measure_frame',side_effect=slow):
            p.receive({'result':self.generated()}); self.pump(entered.is_set)
            p.settings['hole_fraction'].setValue(.2)
            release.set(); self.wait(w)
        self.assertIsNone(w.result); self.assertIsNone(p.measurement)
        self.assertFalse(w.export_button.isEnabled())
        self.assertIn('未应用',p.distance_note.text())

    def test_saved_project_does_not_restart_measurement_or_reset_zoom(self):
        w,p=self.window(); p.receive({'result':self.generated()}); self.wait(w)
        measured=w.result; p.view.scale(1.6,1.6); transform=p.view.transform()
        with temporary_directory() as path, patch.object(QtWidgets.QFileDialog,'getExistingDirectory',return_value=path):
            p.save_project(); self.wait(w)
        self.assertIs(w.result,measured)
        self.assertEqual(p.view.transform(),transform)

    def test_close_cancels_generation_and_measurement_workers(self):
        w,p=self.window(); p.receive({'result':self.generated()}); self.wait(w); w.show()
        def slow(progress,cancel):
            while not cancel(): time.sleep(.005)
            raise InterruptedError('cancelled test generation')
        p.launch(slow,'slow test'); w.close()
        self.pump(lambda:p.thread is None and w.thread is None)
        self.app.processEvents(); self.assertFalse(w.isVisible())

    def test_failed_new_input_restores_last_source_and_result(self):
        w,p=self.window(); p.receive({'result':self.generated()}); self.wait(w)
        old_frame=w.frame.depth_m.copy()
        p.rgb_path.setText('missing-new-image.jpg')
        def fail(progress,cancel):
            raise ValueError('failed new source fixture')
        p.launch(fail,'failed load'); self.wait(w)
        self.assertEqual(p.rgb_path.text(),p.result.scene.source_rgb)
        self.assertFalse(p.dirty)
        self.assertIsNotNone(w.result)
        np.testing.assert_array_equal(old_frame,w.frame.depth_m)
        self.assertIn('failed new source fixture',p.log.toPlainText())

    def test_cancel_removes_latest_pending_measurement(self):
        w,p=self.window(); entered=threading.Event(); release=threading.Event(); calls=[]
        def slow(frame,ds,k,cfg,**kwargs):
            calls.append(frame.metadata['test_tag']); entered.set(); release.wait(5)
            return measure_frame(frame,ds,k,cfg)
        with patch('icicle_measure.gui.measure_frame',side_effect=slow):
            p.receive({'result':self.generated(tag=1)}); self.pump(entered.is_set)
            p.receive({'result':self.generated(1.35,tag=2)})
            w.cancel_measurement(); release.set(); self.wait(w)
        self.assertEqual(calls,[1])
        self.assertIsNone(w._pending_pseudo)
        self.assertIsNone(w.result)
        self.assertEqual(w.frame.metadata['test_tag'],2)
        self.assertIn('已取消',p.distance_note.text())


class DistanceRenderTests(unittest.TestCase):
    def test_current_pair_uses_core_length_and_rejected_pairs_have_no_length(self):
        s=generate('clean'); measured=measure_frame(s.frame,s.detections,s.intrinsics)
        obj=measured.objects[0]; self.assertTrue(verified_distance(obj))
        self.assertIn('%.2f mm'%(obj.distance_m*1000),distance_summary(obj,True,'observed'))
        corrupted=copy.deepcopy(obj); corrupted.endpoints[0].status='rejected'
        self.assertFalse(verified_distance(corrupted))
        self.assertIn('尖端未通过',distance_summary(corrupted))
        self.assertNotIn('%.2f mm'%(obj.distance_m*1000),distance_summary(corrupted))
        for index in (0,1):
            bad=copy.deepcopy(obj); bad.endpoints[index].xyz_m=None
            self.assertFalse(verified_distance(bad))

    def test_annotation_is_selected_only_uses_instance_id_and_clips_at_edges(self):
        s=generate('multi'); result=measure_frame(s.frame,s.detections,s.intrinsics)
        target=s.detections[0]
        image=overlay(s.frame,s.detections,result,selected=(target.id,0),distance_scope='selected')
        self.assertEqual(image.shape,s.frame.rgb.shape)
        base=overlay(s.frame,s.detections,result,distances=False,selected=(target.id,0))
        self.assertTrue(np.any(base!=image))
        edge=generate('edge')
        painted=draw_selected_distance(edge.frame.rgb.copy(),edge.detections[0])
        self.assertEqual(painted.shape,edge.frame.rgb.shape)
