import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PyQt5 import QtWidgets, QtCore, QtGui, QtTest
from icicle_measure.gui import MainWindow, ConfigDialog, configure_font
from icicle_measure import Config
from icicle_measure.simulator import generate
from icicle_measure.io import save_simulation
from tests.helpers import temporary_directory


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_font(cls.app)

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.window.thread:
            self.window.cancel_measurement()
            self.wait_worker()
        self.window.close()
        self.app.processEvents()

    def wait_worker(self):
        deadline = time.monotonic() + 15
        while self.window.thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QtTest.QTest.qWait(10)
        self.assertIsNone(self.window.thread, 'worker did not finish')
        self.app.processEvents()

    def test_import_compute_select_config_export(self):
        with temporary_directory() as directory:
            root = Path(directory)
            manifest = save_simulation(root / 'input', generate('multi'))
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(manifest), 'JSON')):
                QtTest.QTest.mouseClick(self.window.load_button, QtCore.Qt.LeftButton)
            QtTest.QTest.mouseClick(self.window.run_button, QtCore.Qt.LeftButton)
            self.wait_worker()
            self.assertEqual(self.window.table.rowCount(), 5)
            self.assertEqual(self.window.result.objects[0].status, 'accepted')
            self.window.table.selectRow(3)
            self.app.processEvents()
            self.assertIn('ice-2', self.window.details.toPlainText())
            self.assertEqual(self.window.plot.endpoint.name, 'base')
            final_record = self.window.plot.endpoint.history[-1]
            self.assertEqual(self.window.plot.scatter.sample_count, len(final_record.valid_ids))
            self.assertEqual(self.window.plot.scatter.cluster_counts['target'], final_record.target_count)
            self.assertGreaterEqual(final_record.target_count, 80)
            titles = [self.window.plot.tabs.tabText(index)
                      for index in range(self.window.plot.tabs.count())]
            self.assertEqual(titles[:4], ['3D 深度点云', '深度簇分布', '深度与均值变化', '扩窗样本量'])
            QtTest.QTest.mouseClick(self.window.depth_check, QtCore.Qt.LeftButton)
            with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=str(root / 'export')):
                QtTest.QTest.mouseClick(self.window.export_button, QtCore.Qt.LeftButton)
            saved = json.loads((root / 'export/measurement.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['objects'][0]['distance_m'], self.window.result.objects[0].distance_m)
            screenshot_dir = Path(__file__).resolve().parents[1] / 'outputs/rgbd_validation'
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            for index, suffix in enumerate(('terrain', 'samples', 'histogram', 'counts')):
                self.window.plot.tabs.setCurrentIndex(index)
                self.app.processEvents()
                self.assertTrue(self.window.grab().save(str(screenshot_dir / ('gui_preview_' + suffix + '.png'))))
                if index == 0:
                    self.assertTrue(self.window.grab().save(str(screenshot_dir / 'gui_preview.png')))
            self.window.plot.tabs.setCurrentIndex(0)
            dialog = ConfigDialog(self.window.config, self.window)
            dialog.editor.setPlainText(json.dumps({'confidence_min': .9}))
            dialog.accept()
            self.assertEqual(dialog.value.confidence_min, .9)
            self.window.set_config(dialog.value)
            self.assertIsNone(self.window.result)
            self.assertFalse(self.window.export_button.isEnabled())
            QtTest.QTest.mouseClick(self.window.run_button, QtCore.Qt.LeftButton)
            self.wait_worker()
            self.assertEqual(self.window.result.config['confidence_min'], .9)

    def test_real_thread_cancel_and_rerun(self):
        from icicle_measure.models import Detection
        s = generate('drift')
        repeated = [Detection(str(i), s.detections[0].bbox, s.detections[0].keypoints) for i in range(300)]
        self.window.set_input(s.frame, repeated, s.intrinsics)
        QtTest.QTest.mouseClick(self.window.run_button, QtCore.Qt.LeftButton)
        self.assertFalse(self.window.load_button.isEnabled())
        QtTest.QTest.mouseClick(self.window.cancel_button, QtCore.Qt.LeftButton)
        self.wait_worker()
        self.assertTrue(self.window.result.cancelled)
        self.assertTrue(self.window.run_button.isEnabled())
        self.window.generate_simulation()
        self.window.start_measurement()
        self.wait_worker()
        self.assertFalse(self.window.result.cancelled)
        self.assertEqual(self.window.result.objects[0].status, 'accepted')

    def test_worker_exception_releases_ui(self):
        self.window.generate_simulation()
        self.window.frame.depth_m = self.window.frame.depth_m[:2]
        errors = []
        self.window.report_error = errors.append
        self.window.start_measurement()
        self.wait_worker()
        self.assertTrue(errors)
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())

    def test_separate_file_import_and_config_roundtrip(self):
        with temporary_directory() as directory:
            root = Path(directory)
            manifest = save_simulation(root / 'input', generate('holes'))
            paths = [root / 'input/rgb.png', root / 'input/depth.npy', root / 'input/detections.json', manifest]
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', side_effect=[(str(p), '') for p in paths]):
                QtTest.QTest.mouseClick(self.window.files_button, QtCore.Qt.LeftButton)
            self.assertEqual(len(self.window.detections), 1)
            self.window.set_config(Config(confidence_min=.9))
            with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=(str(root / 'config.json'), '')):
                QtTest.QTest.mouseClick(self.window.config_save_button, QtCore.Qt.LeftButton)
            self.window.set_config(Config())
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(root / 'config.json'), '')):
                QtTest.QTest.mouseClick(self.window.config_load_button, QtCore.Qt.LeftButton)
            self.assertEqual(self.window.config.confidence_min, .9)
            self.window.generate_simulation()
            with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=str(root / 'saved_sim')):
                QtTest.QTest.mouseClick(self.window.save_sim_button, QtCore.Qt.LeftButton)
            self.assertTrue((root / 'saved_sim/input.json').exists())


if __name__ == '__main__':
    unittest.main()
