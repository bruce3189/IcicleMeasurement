"""PyQt5 front end; numerical work remains in the independent core."""
import json
import os
import sys
import threading
import copy
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from .models import Config
from .core import measure_frame
from .simulator import generate, SCENARIOS
from .geometry import build_cloud, project as project_points
from .io import (load_bundle, load_files, export_results, save_simulation,
                 write_json, read_json, depth_quality, save_input_bundle)
from .render import overlay, distance_summary, verified_distance

from .ui.common import STATUS, configure_font, ImageView, ConfigDialog
from .ui.charts import PlotCanvas, HistogramCanvas, WindowCountsCanvas
from .ui.cloud_canvas import DepthTerrainCanvas
from .ui.cloud_panel import DepthTerrainPanel
from .ui.diagnostics import MeasurementCriteriaPanel, DiagnosticsPlot


class Worker(QtCore.QObject):
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal()

    def __init__(self, frame, detections, intrinsics, config, stop):
        super().__init__()
        self.args = (frame, detections, intrinsics, config)
        self.stop = stop

    @QtCore.pyqtSlot()
    def run(self):
        try:
            result = measure_frame(*self.args, cancel=self.stop.is_set, progress=self.progress.emit)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class PoseMeasurementWorker(Worker):
    def __init__(self, packet, pose_config, config, stop, detector=None):
        super().__init__(packet.frame, [], packet.intrinsics, config, stop)
        self.packet, self.pose_config, self.detector = packet, pose_config, detector

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from .pose import PoseDetector
            detector = self.detector
            if detector is None or detector.config != self.pose_config:
                detector = PoseDetector(self.pose_config)
            detections = [] if self.stop.is_set() else detector.detect(self.packet.frame, self.packet.sequence)
            self.packet.frame.metadata['pose'] = asdict(self.pose_config)
            result = measure_frame(self.packet.frame, detections, self.packet.intrinsics,
                                   self.args[3], cancel=self.stop.is_set, progress=self.progress.emit)
            completed_at = datetime.now(timezone.utc).isoformat()
            result.source['completed_at'] = completed_at
            self.completed.emit((self.packet, detections, result, detector, completed_at))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('冰凌双端点 · RGB-D 测距工作台')
        self.resize(1450, 940)
        self.frame = self.intrinsics = self.result = self.simulation = None
        self.detections = []
        self.config = Config()
        self.thread = self.worker = None
        self.stop = threading.Event()
        self.row_map = []
        self.close_pending = False
        self.camera_dialog = None
        self.pseudo_dialog = None
        self.pose_detector = None
        self.selected_cloud_sample = None
        self._input_revision = 0
        self._pending_pseudo = None
        self._cancelled_pseudo_input = None
        self._measurement_cancelled = False
        self._pseudo_active = False
        self._pseudo_binding = None
        self._target_sync = False
        self._preferred_target_id = None
        self._preferred_endpoint = 0
        self._cloud_expanded = False
        self._expanded_restore = {}
        root = QtWidgets.QWidget()
        root.setObjectName('mainRoot')
        self.workspace = QtWidgets.QTabWidget()
        self.scene_page = QtWidgets.QWidget()
        self.scene_layout = QtWidgets.QVBoxLayout(self.scene_page)
        self.scene_layout.setContentsMargins(0, 0, 0, 0)
        self.analysis_page = root
        self.workspace.addTab(self.scene_page, '场景生成')
        self.workspace.addTab(root, '测距分析')
        self.workspace.setCurrentIndex(1)
        self.setCentralWidget(self.workspace)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        title = QtWidgets.QLabel('冰凌双端点测距  /  RGB-D')
        title.setStyleSheet('font-size:22px; font-weight:600; padding:8px;')
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(title); heading.addStretch()
        self.pseudo_button = self.button(heading, 'RGB 生成伪深度', self.open_pseudo_depth)
        layout.addLayout(heading)
        self.input_note = QtWidgets.QLabel('场景假设：冰凌在背景前方，邻域目标深度可代表端点。输出为估计值；评分是规则满足度，不是正确概率。')
        self.input_note.setWordWrap(True)
        self.input_note.setStyleSheet('color:#876315; background:#fff4d8; padding:8px;')
        layout.addWidget(self.input_note)
        toolbar = QtWidgets.QHBoxLayout()
        layout.addLayout(toolbar)
        self.load_button = self.button(toolbar, '导入输入包', self.choose_bundle)
        self.files_button = self.button(toolbar, '分别导入文件', self.choose_files)
        self.scenario = QtWidgets.QComboBox()
        for key, name in SCENARIOS.items():
            self.scenario.addItem(name, key)
        self.scenario.setCurrentIndex(1)
        toolbar.addWidget(self.scenario)
        self.sim_button = self.button(toolbar, '生成模拟', self.generate_simulation)
        self.run_button = self.button(toolbar, '开始计算', self.start_measurement)
        self.cancel_button = self.button(toolbar, '取消', self.cancel_measurement)
        self.export_button = self.button(toolbar, '导出结果', self.choose_export)
        self.camera_button = self.button(toolbar, 'D435i 相机', self.open_camera)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        toolbar.addStretch()
        self.depth_mode = QtWidgets.QComboBox()
        self.depth_mode.addItem('近簇均值近似（推荐）', 'near_cluster_mean')
        self.depth_mode.addItem('严格局部一致', 'strict_local')
        self.depth_mode.setToolTip('保留原关键点像素，用可信近簇算术均值估深。近似模式下，平面偏差和新增区域均值差仅提示。')
        self.depth_mode.currentIndexChanged.connect(self.change_depth_mode)
        toolbar.addWidget(self.depth_mode)
        options = QtWidgets.QHBoxLayout()
        layout.addLayout(options)
        self.depth_check = QtWidgets.QCheckBox('显示深度')
        self.box_check = QtWidgets.QCheckBox('检测框/编号')
        self.point_check = QtWidgets.QCheckBox('关键点/连线')
        self.distance_check = QtWidgets.QCheckBox('距离')
        self.cluster_check = QtWidgets.QCheckBox('所选端点采样簇')
        for cb in (self.depth_check, self.box_check, self.point_check, self.distance_check, self.cluster_check):
            cb.setChecked(cb is not self.depth_check)
            cb.toggled.connect(self.refresh_image)
            options.addWidget(cb)
        self.params_button = self.button(options, '参数设置', self.edit_config)
        self.config_load_button = self.button(options, '加载参数', self.load_config)
        self.config_save_button = self.button(options, '保存参数', self.save_config)
        self.save_sim_button = self.button(options, '保存模拟输入', self.choose_save_simulation)
        self.save_frame_button = self.button(options, '保存当前 RGB-D', self.choose_save_frame)
        options.addStretch()
        self._analysis_toolbar_layouts = (heading,toolbar,options)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, 1)
        self.image = ImageView()
        split.addWidget(self.image)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['目标', '端点', '状态', 'Z (mm)', '估计 XYZ (m)', '评分', '端点欧氏距离 (mm)', '判定/拒绝原因'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        for column, width in enumerate((78, 58, 58, 82, 205, 55, 96)):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setMinimumHeight(138)
        right_layout.addWidget(self.table, 1)
        self.distance_note = QtWidgets.QLabel('端点估计欧氏距离：待计算')
        self.distance_note.setWordWrap(True)
        self.distance_note.setStyleSheet('font-weight:600;color:#245c9b;padding:4px;')
        right_layout.addWidget(self.distance_note)
        self.plot = DiagnosticsPlot()
        self.plot.terrain.expand_requested.connect(self.expand_pointcloud)
        self.plot.tabs.currentChanged.connect(self.update_compact_summary)
        self.plot.terrain.canvas.sample_selected.connect(self.show_cloud_sample)
        right_layout.addWidget(self.plot, 3)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(140)
        detail_tab = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_tab)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.addWidget(self.details)
        self.plot.tabs.addTab(detail_tab, '判定记录')
        split.addWidget(right)
        split.setSizes([620, 800])
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.statusBar().showMessage('导入一组 RGB + 深度 + 标注 + 内参，或生成模拟场景。滚轮缩放，拖动平移，双击适配图像。')
        self.setStyleSheet('''
            QMainWindow, QWidget#mainRoot {background:#eef2f7;color:#253449;}
            QLabel {color:#26364a;}
            QPushButton {background:#ffffff;border:1px solid #d3dce7;border-radius:7px;padding:7px 11px;}
            QPushButton:hover {background:#eaf1fb;border-color:#9db5d3;}
            QPushButton:pressed {background:#dce8f7;}
            QPushButton:disabled {color:#9ca8b7;background:#e8edf3;}
            QComboBox {background:#fff;border:1px solid #d3dce7;border-radius:6px;padding:6px;}
            QTableWidget {background:#fff;border:1px solid #dce4ed;border-radius:8px;gridline-color:#edf1f5;}
            QHeaderView::section {background:#f3f6fa;color:#58677b;border:0;border-bottom:1px solid #dce4ed;padding:7px;}
            QTabWidget::pane {border:1px solid #dce4ed;border-radius:8px;background:#f4f7fb;}
            QTabBar::tab {background:#e7ecf3;color:#66758a;padding:7px 13px;margin-right:3px;border-top-left-radius:6px;border-top-right-radius:6px;}
            QTabBar::tab:selected {background:#fff;color:#245c9b;font-weight:600;}
            QPlainTextEdit {background:#fff;border:1px solid #dce4ed;border-radius:8px;padding:7px;}
            QProgressBar {border:0;border-radius:4px;background:#dfe6ef;text-align:center;}
            QProgressBar::chunk {background:#36a582;border-radius:4px;}
        ''')
        self.workspace.currentChanged.connect(lambda index: self.open_pseudo_depth() if index == 0 else None)
        self._compact_mode = None
        self.apply_compact_layout()
        self._escape_cloud = QtWidgets.QShortcut(QtGui.QKeySequence('Esc'),self)
        self._escape_cloud.activated.connect(lambda:self.plot.terrain.expand_button.setChecked(False) if self._cloud_expanded else None)

    def expand_pointcloud(self, expanded):
        if expanded == self._cloud_expanded:
            return
        self._cloud_expanded = expanded
        terrain = self.plot.terrain
        terrain.expand_button.setText('收起点云 Esc' if expanded else '展开点云')
        if expanded:
            self.workspace.setCurrentIndex(1)
            self._expanded_tab = self.plot.tabs.currentWidget()
            self._expanded_compact = self._compact_mode
            self.plot.tabs.setCurrentWidget(terrain)
            widgets = [self.image,self.table,self.distance_note,self.input_note,self.progress_bar,
                       self.plot.guide,self.plot.tabs.tabBar()]
            def collect(layout):
                for i in range(layout.count()):
                    item=layout.itemAt(i)
                    if item.widget() is not None:
                        widgets.append(item.widget())
                    elif item.layout() is not None:
                        collect(item.layout())
            for layout in self._analysis_toolbar_layouts:
                collect(layout)
            self._expanded_restore = {widget:not widget.isHidden() for widget in widgets}
            for widget in self._expanded_restore:
                widget.hide()
            terrain.canvas.setFocus()
            self.statusBar().showMessage('点云展开查看：滚轮缩放，Shift 精调，选点后 F 聚焦；Esc 收起。')
        else:
            for widget,visible in self._expanded_restore.items():
                widget.setVisible(visible)
            self._expanded_restore = {}
            self.plot.tabs.setCurrentWidget(self._expanded_tab)
            if (self.height()<900) != self._expanded_compact:
                self._compact_mode = None
                self.apply_compact_layout()

    def apply_compact_layout(self):
        if self._cloud_expanded:
            return
        compact = self.height() < 900
        if compact != self._compact_mode:
            self._compact_mode = compact
            self.plot.guide.setVisible(not compact)
            self.plot.terrain.settings_scroll.setMaximumHeight(110 if compact else 180)
            self.plot.terrain.settings_toggle.setChecked(not compact)
        self.update_compact_summary()

    def update_compact_summary(self, *_):
        if not self._cloud_expanded:
            self.distance_note.setVisible(not self._compact_mode or self.plot.tabs.currentWidget() is not self.plot.terrain)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_compact_mode'):
            self.apply_compact_layout()

    def button(self, layout, text, callback):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def report_error(self, message):
        self.statusBar().showMessage('错误：' + message)
        QtWidgets.QMessageBox.warning(self, '操作未完成', message)

    def set_input(self, frame, detections, intrinsics, source='输入文件'):
        frame.validate()
        intrinsics.validate()
        self._input_revision += 1
        self._pending_pseudo = None
        self._pseudo_active = False
        self.frame, self.detections, self.intrinsics = frame, detections, intrinsics
        if self.pseudo_dialog:
            self.pseudo_dialog.clear_measurement('分析输入已更新，等待当前输入的测距结果')
        self.selected_cloud_sample = None
        self.result = None
        self.table.setRowCount(0)
        self.row_map = []
        self.plot.set_data(None, None)
        pseudo = frame.metadata.get('is_synthetic') is True
        geometry = ('模型伪深度 / 人工合成；假设针孔几何，原照片畸变未经标定，非实测深度。' if pseudo
                    else '输入已声明对齐、去畸变，深度为光轴 Z。')
        self.input_note.setText(('模型伪深度 / 人工合成 · 基于伪深度的估计距离；人工设定尺度、假设或用户提供内参。长度先验不构成测量真值。' if pseudo
            else '场景假设：冰凌在背景前方，邻域目标深度可代表端点。输出为估计值；评分是规则满足度，不是正确概率。'))
        self.table.setHorizontalHeaderItem(6, QtWidgets.QTableWidgetItem('伪深度端点欧氏距离 (mm)' if pseudo else '端点欧氏距离 (mm)'))
        self.update_distance_summary()
        self.details.setPlainText('输入：%s\n图像：%d × %d\n目标数：%d\n相机内参：%s\n%s' % (
            source, frame.rgb.shape[1], frame.rgb.shape[0], len(detections), asdict(intrinsics), geometry))
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.refresh_image(fit=True)
        quality = depth_quality(frame)
        frame.metadata['depth_quality'] = quality
        unit = frame.metadata.get('depth_unit', 'm（计算层）')
        quality_text = '有效深度 %.1f%% · 原单位 %s' % (quality['valid_fraction'] * 100, unit)
        if 'depth_scale_m_per_unit' in frame.metadata:
            quality_text += ' · 每单位 %.8g m' % frame.metadata['depth_scale_m_per_unit']
        if quality['valid_fraction'] < .1:
            quality_text += ' · 有效覆盖较低，请检查单位、对齐与原始数据'
        self.details.appendPlainText('\n' + quality_text + '\n来源：' + json.dumps(frame.metadata, ensure_ascii=False))
        self.statusBar().showMessage(source + ' · ' + quality_text + '，等待计算。')

    def open_pseudo_depth(self):
        if self.pseudo_dialog is None:
            try:
                from .pseudo_gui import PseudoDepthPanel
            except ImportError as exc:
                self.report_error('可选伪深度组件依赖缺失：' + str(exc))
                return
            self.pseudo_dialog = PseudoDepthPanel(self.scene_page)
            self.scene_layout.addWidget(self.pseudo_dialog)
            self.pseudo_dialog.measurement_requested.connect(self.receive_pseudo_input)
            self.pseudo_dialog.scene_ready.connect(self.queue_pseudo_input)
            self.pseudo_dialog.measurement_invalidated.connect(self.invalidate_pseudo_measurement)
            self.pseudo_dialog.target_selected.connect(self.select_generation_target)
            self.pseudo_dialog.task_stopped.connect(lambda: QtCore.QTimer.singleShot(0, self.close)
                if self.close_pending else None)
        self.workspace.setCurrentIndex(0)
        if self.pseudo_dialog.result is None and self.pseudo_dialog.thread is None:
            self.pseudo_dialog.choose_sample(self.pseudo_dialog.samples.currentText())

    def receive_pseudo_input(self, frame, detections, intrinsics):
        if self.close_pending:
            return
        binding = (id(self.pseudo_dialog.result), self.pseudo_dialog.variant.currentData()) if self.pseudo_dialog else None
        if not (self._pseudo_active and binding is not None and binding == self._pseudo_binding
                and (self.thread is not None or self.result is not None)):
            self.queue_pseudo_input(frame, detections, intrinsics)
        self.workspace.setCurrentIndex(1)

    def invalidate_pseudo_measurement(self, reason='测距结果待更新'):
        # Editing an unrelated generator page must not invalidate a real-camera input.
        if not self._pseudo_active and self._pending_pseudo is None:
            return
        self._input_revision += 1
        self._pending_pseudo = None
        self._cancelled_pseudo_input = None
        if self.thread is not None:
            self.stop.set()
        self.result = None
        self.table.setRowCount(0); self.row_map = []
        self.plot.set_data(None, None)
        self.export_button.setEnabled(False)
        self.selected_cloud_sample = None
        self.details.setPlainText(reason)
        self.update_distance_summary(reason=reason)
        self.refresh_image()
        self.statusBar().showMessage(reason)

    def queue_pseudo_input(self, frame, detections, intrinsics):
        if self.close_pending:
            return
        if self.camera_dialog:
            self.camera_dialog.continuous.setChecked(False)
        self._input_revision += 1
        self._measurement_cancelled = False
        # Snapshot all inputs: edits in the other page cannot mutate an active worker.
        binding = (id(self.pseudo_dialog.result), self.pseudo_dialog.variant.currentData()) if self.pseudo_dialog else None
        target = self.pseudo_dialog.object_choice.currentData() if self.pseudo_dialog else None
        packet = (copy.deepcopy(frame), copy.deepcopy(detections), copy.deepcopy(intrinsics), binding, target)
        self._pseudo_active = True
        self.invalidate_pseudo_measurement('正在自动测距；旧距离已失效。')
        if self.thread is not None:
            self._pending_pseudo = packet
            return
        self.commit_pseudo_input(packet)

    def commit_pseudo_input(self, packet, start=True):
        frame, detections, intrinsics, binding, target = packet
        self.simulation = None
        self.set_input(frame, detections, intrinsics, '模型伪深度 / 人工合成 · ' + frame.metadata.get('depth_variant', ''))
        self._pseudo_active = True
        self._pseudo_binding = binding
        self._preferred_target_id = target
        if self.pseudo_dialog:
            self.pseudo_dialog.clear_measurement('正在自动测距…')
        if start:
            self.start_measurement()
        else:
            self.update_distance_summary(reason='已取消，待重新计算')
            if self.pseudo_dialog:
                self.pseudo_dialog.clear_measurement('已取消，待重新计算')

    def select_generation_target(self, target):
        if self._target_sync or not self._pseudo_active:
            return
        self._preferred_target_id = target
        current, _ = self.selected()
        if current is not None and current.id == target:
            return
        for row, (obj, index) in enumerate(self.row_map):
            if obj.id == target:
                self._target_sync = True
                self.table.selectRow(row)
                self._target_sync = False
                break

    def update_distance_summary(self, obj=None, reason='待计算'):
        pseudo = self.frame is not None and self.frame.metadata.get('input_kind') == 'pseudo_rgbd'
        variant = self.frame.metadata.get('depth_variant','') if self.frame is not None else ''
        text = distance_summary(obj, pseudo, variant, reason)
        self.distance_note.setText(text)
        self.plot.terrain.distance_note.setText(text)

    def open_camera(self):
        if self.camera_dialog is None:
            from .live_gui import CameraDialog
            self.camera_dialog = CameraDialog(lambda: self.thread is None and not self.close_pending, self)
            self.camera_dialog.measure_requested.connect(self.start_pose_measurement)
        self.camera_dialog.show()
        self.camera_dialog.raise_()

    def start_pose_measurement(self, packet, pose_config):
        if self.thread is not None or self.close_pending:
            return
        self.simulation = None
        self.set_input(packet.frame, [], packet.intrinsics, '相机冻结帧 #%d · %s' % (packet.sequence, packet.captured_at))
        self.stop = threading.Event()
        self.set_busy(True)
        self.details.appendPlainText('\n正在对本帧运行两点 YOLO11 和深度测距…')
        self.statusBar().showMessage('正在检测并测量相机帧 #%d，结果将绑定此帧。' % packet.sequence)
        worker = PoseMeasurementWorker(packet, pose_config, self.config, self.stop, self.pose_detector)
        self.launch_worker(worker, self.receive_pose_result, self.report_pose_error)

    def receive_pose_result(self, payload):
        packet, detections, result, self.pose_detector, completed_at = payload
        self.frame, self.detections, self.intrinsics = packet.frame, detections, packet.intrinsics
        self.receive_result(result)
        if self.camera_dialog:
            self.camera_dialog.processing_finished(packet.sequence, completed_at)

    def report_pose_error(self, message):
        self.statusBar().showMessage('检测/测距失败：' + message)
        self.details.appendPlainText('\n失败：' + message)
        if self.camera_dialog:
            self.camera_dialog.processing_failed(message)

    def choose_save_frame(self):
        if self.frame is None:
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '保存当前冻结 RGB-D 输入（同名文件会覆盖）')
        if path:
            try:
                manifest = save_input_bundle(path, self.frame, self.detections, self.intrinsics)
                self.statusBar().showMessage('已保存可重放的输入包：' + str(manifest))
            except Exception as exc:
                self.report_error(str(exc))

    def import_bundle(self, path):
        self.set_input(*load_bundle(path), source=str(path))
        self.simulation = None

    def choose_bundle(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择 input.json 输入包', '', 'JSON (*.json)')
        if path:
            try:
                self.import_bundle(path)
            except Exception as exc:
                self.report_error(str(exc))

    def choose_files(self):
        paths = []
        for name, pattern in [('RGB 图', 'Images (*.png *.jpg *.jpeg *.bmp)'),
                              ('深度图（不能是彩色深度预览图）', 'Depth (*.npy *.png *.tiff *.tif)'),
                              ('检测框与关键点', 'Labels (*.json *.txt)'),
                              ('内参与深度元数据', 'JSON (*.json)')]:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择' + name, '', pattern)
            if not path:
                return
            paths.append(path)
        try:
            self.set_input(*load_files(*paths), source=paths[0])
            self.simulation = None
        except Exception as exc:
            self.report_error(str(exc))

    def generate_simulation(self):
        self.simulation = generate(self.scenario.currentData())
        s = self.simulation
        self.set_input(s.frame, s.detections, s.intrinsics, '模拟：' + SCENARIOS[s.scenario] + '（seed=42）')

    def set_config(self, config):
        config.validate()
        self.config = config
        self.depth_mode.blockSignals(True)
        self.depth_mode.setCurrentIndex(self.depth_mode.findData(config.endpoint_depth_mode))
        self.depth_mode.blockSignals(False)
        self._input_revision += 1
        if self.thread is not None:
            self.stop.set()
        self.result = None
        self.table.setRowCount(0)
        self.row_map = []
        self.plot.set_data(None, None)
        self.export_button.setEnabled(False)
        self.details.setPlainText('参数已修改；旧结果已清除，请重新计算。')
        self.refresh_image()
        self.statusBar().showMessage('参数已修改，请重新计算。')
        self.update_distance_summary(reason='参数已修改，待重新计算')
        if self._pseudo_active and self.pseudo_dialog:
            self.pseudo_dialog.clear_measurement('参数已修改，正在自动重算…')
            if self.pseudo_dialog.thread is None:
                self.pseudo_dialog.emit_scene()

    def change_depth_mode(self, *_):
        self.set_config(replace(self.config, endpoint_depth_mode=self.depth_mode.currentData()))

    def edit_config(self):
        dialog = ConfigDialog(self.config, self)
        if dialog.exec_() == dialog.Accepted:
            self.set_config(dialog.value)

    def load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '加载参数', '', 'JSON (*.json)')
        if path:
            try:
                self.set_config(Config.from_dict(read_json(path)))
            except Exception as exc:
                self.report_error(str(exc))

    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存参数', 'config.json', 'JSON (*.json)')
        if path:
            try:
                write_json(path, asdict(self.config))
            except Exception as exc:
                self.report_error(str(exc))

    def set_busy(self, busy):
        for widget in (self.load_button, self.files_button, self.sim_button, self.scenario,
                       self.params_button, self.config_load_button, self.save_sim_button, self.save_frame_button,
                       self.depth_mode):
            widget.setEnabled(not busy)
        self.run_button.setEnabled(not busy and self.frame is not None)
        self.cancel_button.setEnabled(busy)
        self.export_button.setEnabled(not busy and self.result is not None)

    def start_measurement(self):
        if self.frame is None or self.thread is not None:
            return
        if self._pseudo_active and self.pseudo_dialog and self.pseudo_dialog.dirty:
            self.statusBar().showMessage('请先应用生成页修正，再进行测距。')
            return
        self.stop = threading.Event()
        self._measurement_cancelled = False
        self.result = None
        self.table.setRowCount(0)
        self.row_map = []
        self.plot.set_data(None, None)
        self.details.setPlainText('正在计算…')
        self.update_distance_summary(reason='正在计算…')
        if self._pseudo_active and self.pseudo_dialog:
            self.pseudo_dialog.clear_measurement('正在自动测距…')
        self.progress_bar.setRange(0, max(1, len(self.detections)))
        self.progress_bar.setValue(0)
        self.set_busy(True)
        self.refresh_image()
        self.statusBar().showMessage('正在逐框计算两个端点…')
        args = copy.deepcopy((self.frame, self.detections, self.intrinsics, self.config))
        self.launch_worker(Worker(*args, self.stop), self.receive_result, self.measurement_error)

    def measurement_error(self, message):
        self.update_distance_summary(reason='测距失败')
        if self._pseudo_active and self.pseudo_dialog:
            self.pseudo_dialog.clear_measurement('测距失败：' + message)
            self.details.setPlainText('测距失败：' + message)
            self.statusBar().showMessage('测距失败：' + message)
        else:
            self.report_error(message)

    def launch_worker(self, worker, receiver, error_handler):
        revision = self._input_revision
        self.thread = QtCore.QThread(self)
        self.worker = worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(lambda value: receiver(value) if revision == self._input_revision and not self.close_pending else None)
        self.worker.failed.connect(lambda message: error_handler(message) if revision == self._input_revision and not self.close_pending else None)
        self.worker.progress.connect(lambda n, total: self.progress_bar.setValue(n) if revision == self._input_revision else None)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.worker_stopped)
        self.thread.start()

    def cancel_measurement(self):
        self._measurement_cancelled = True
        self._cancelled_pseudo_input = self._pending_pseudo
        self._pending_pseudo = None
        if self.camera_dialog:
            self.camera_dialog.continuous.setChecked(False)
        self.stop.set()
        self.cancel_button.setEnabled(False)
        self.statusBar().showMessage('正在取消；等待当前窗口结束…')

    def worker_stopped(self):
        thread = self.thread
        self.thread = self.worker = None
        if thread:
            thread.deleteLater()
        self.set_busy(False)
        if self.close_pending:
            self.close()
        elif self._pending_pseudo is not None:
            packet, self._pending_pseudo = self._pending_pseudo, None
            self.commit_pseudo_input(packet)
        elif self._cancelled_pseudo_input is not None:
            packet, self._cancelled_pseudo_input = self._cancelled_pseudo_input, None
            self.commit_pseudo_input(packet, start=False)
        elif self._measurement_cancelled and self.result is None:
            self.update_distance_summary(reason='已取消，待重新计算')
            if self._pseudo_active and self.pseudo_dialog:
                self.pseudo_dialog.clear_measurement('已取消，待重新计算')

    def receive_result(self, result):
        self.result = result
        self.row_map = []
        self.table.setRowCount(0)
        for obj in result.objects:
            for index, ep in enumerate(obj.endpoints or [None]):
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.row_map.append((obj, index if ep else None))
                xyz = ep.xyz_m if ep else None
                xyz_text = '—' if xyz is None else '(%.3f, %.3f, %.3f)' % xyz
                cells = [obj.id, ('尖端' if index == 0 else '根部') if ep else '—',
                         STATUS[ep.status if ep else obj.status],
                         '%.2f' % (ep.depth_m * 1000) if ep and ep.depth_m is not None else '—',
                         xyz_text,
                         '%.3f' % ep.confidence if ep else '—',
                         '%.2f' % (obj.distance_m * 1000) if verified_distance(obj) else '—',
                         ep.reason if ep else obj.reason]
                for col, text in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setToolTip(text)
                    self.table.setItem(row, col, item)
        accepted = sum(o.status == 'accepted' for o in result.objects)
        self.statusBar().showMessage(('已取消，保留已完成部分。' if result.cancelled else '完成。') +
                                    '测距通过 %d / %d 个目标。' % (accepted, len(self.detections)))
        if self._pseudo_active and self.pseudo_dialog:
            self.pseudo_dialog.set_measurement_result(result)
        if self.table.rowCount():
            preferred = next((row for row,(obj,index) in enumerate(self.row_map)
                              if obj.id == self._preferred_target_id and index == self._preferred_endpoint), 0)
            self.table.selectRow(preferred)
        else:
            self.details.setPlainText('未完成任何目标。' if result.cancelled else '输入中没有目标。')
        self.refresh_image()
        if result.source.get('sequence') is not None:
            self.statusBar().showMessage(self.statusBar().currentMessage() + ' 冻结帧 #%s · %s' % (
                result.source['sequence'], result.source.get('captured_at', '')))

    def selected(self):
        row = self.table.currentRow()
        return self.row_map[row] if 0 <= row < len(self.row_map) else (None, None)

    def refresh_image(self, checked=False, fit=False):
        if self.frame is None:
            return
        obj, index = self.selected()
        selected = (obj.id, index) if obj is not None and index is not None else None
        image = overlay(self.frame, self.detections, self.result, self.depth_check.isChecked(),
                        self.box_check.isChecked(), self.point_check.isChecked(), self.distance_check.isChecked(),
                        selected, self.cluster_check.isChecked(), distance_scope='selected')
        if self.selected_cloud_sample is not None:
            import cv2
            pid = self.selected_cloud_sample['pixel_id']
            px, py = pid % image.shape[1], pid // image.shape[1]
            cv2.circle(image, (px, py), 6, (255, 240, 140), 1, cv2.LINE_AA)
        self.image.set_image(image, fit)

    def show_cloud_sample(self, sample):
        self.selected_cloud_sample = sample
        self.refresh_image()

    def show_selected(self):
        self.selected_cloud_sample = None
        obj, index = self.selected()
        self.update_distance_summary(obj)
        if obj is None:
            self.plot.set_data(None, None)
            return
        self._preferred_target_id = obj.id
        self._preferred_endpoint = index if index is not None else 0
        if self._pseudo_active and self.pseudo_dialog and not self._target_sync:
            self._target_sync = True
            target_index = self.pseudo_dialog.object_choice.findData(obj.id)
            if target_index >= 0:
                self.pseudo_dialog.object_choice.setCurrentIndex(target_index)
            self._target_sync = False
        if index is None:
            self.details.setPlainText(obj.reason)
            self.plot.set_data(None, None)
            self.refresh_image()
            return
        ep = obj.endpoints[index]
        lines = ['目标 %s / %s：%s' % (obj.id, '尖端' if index == 0 else '根部', STATUS[ep.status]),
                 ep.reason, '原始像素：%s；估计 XYZ (m)：%s' % (ep.pixel, ep.xyz_m),
                 '评分 %.4f（不是概率）；目标距离：%s m' % (ep.confidence, obj.distance_m)]
        lines.append('来源：' + json.dumps(self.frame.metadata, ensure_ascii=False))
        if self.simulation and obj.id in self.simulation.truth:
            truth = self.simulation.truth[obj.id]
            lines.append('模拟真值 Z=%.6f m，长度=%.6f m' % (truth['depth_m'][index], truth['distance_m']))
            if ep.depth_m is not None:
                lines.append('模拟 Z 误差=%.3f mm' % (abs(ep.depth_m - truth['depth_m'][index]) * 1000))
        lines.append('\n扩窗记录：')
        for r in ep.history:
            fmt = lambda value: '—' if value is None else '%.4f' % value
            lines.append('r=%d | 近/远/目标=%d/%d/%d | 均值=%s m | 分离=%s | 间隔=%s m\n'
                         '  中心变化=%s m | 成员 J=%s | 总分=%.3f | %s' % (
                             r.radius, r.near_count, r.far_count, r.target_count, fmt(r.center_m),
                             fmt(r.separation), fmt(r.gap_m), fmt(r.center_change_m), fmt(r.membership),
                             r.score, '；'.join(r.reasons) or '全部通过'))
            lines.append('  锚区=%s | 内部多群=%s | 梯度偏差=%s mm | 新增目标/远簇=%s/%s | '
                         '覆盖近/远/目标=%s/%s/%s | 连续通过=%d' % (
                             r.anchor.get('count', '—'), r.subgroups.get('ambiguous', '—'),
                             '—' if r.plane.get('bias_m') is None else '%.3f' % (r.plane['bias_m'] * 1000),
                             r.growth.get('new_target', '—'), r.growth.get('new_far', '—'),
                             r.coverage.get('near', '—'), r.coverage.get('far', '—'), r.coverage.get('target', '—'),
                             r.consecutive_passes))
            if r.warnings:
                lines.append('  仅提示（不阻断）：' + '；'.join(r.warnings))
        if ep.history:
            lines.append('\n最后窗口分项评分：' + json.dumps(ep.history[-1].scores, ensure_ascii=False))
        self.details.setPlainText('\n'.join(lines))
        detection = next((item for item in self.detections if item.id == obj.id), None)
        self.plot.set_data(ep, self.frame.depth_m, detection, self.intrinsics, self.config)
        self.refresh_image()

    def export_to(self, path):
        if self.result is None:
            raise ValueError('请先计算')
        return export_results(path, self.frame, self.detections, self.result)

    def choose_export(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择结果保存文件夹（同名结果文件会覆盖）')
        if path:
            try:
                self.export_to(path)
                self.statusBar().showMessage('已导出 JSON、CSV、参数及标注图：' + path)
            except Exception as exc:
                self.report_error(str(exc))

    def choose_save_simulation(self):
        if self.simulation is None:
            self.statusBar().showMessage('请先生成模拟场景。')
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '保存模拟输入（同名文件会覆盖）')
        if path:
            try:
                save_simulation(path, self.simulation)
                self.statusBar().showMessage('模拟 RGB、深度、标注、内参和真值已保存：' + path)
            except Exception as exc:
                self.report_error(str(exc))

    def closeEvent(self, event):
        if self.pseudo_dialog and self.pseudo_dialog.thread is not None:
            self.close_pending = True
            self._pending_pseudo = None
            self.pseudo_dialog.close_pending = True
            self.pseudo_dialog.cancel()
            self.stop.set()
            event.ignore()
            return
        if self.camera_dialog and self.camera_dialog.runner and self.camera_dialog.runner.is_alive():
            self.close_pending = True
            self.camera_dialog.closing = True
            self.camera_dialog.stop_capture()
            if self.thread is not None:
                self.cancel_measurement()
            event.ignore()
            return
        if self.thread is not None:
            self.close_pending = True
            self.cancel_measurement()
            event.ignore()
        else:
            if self.camera_dialog:
                self.camera_dialog.close()
            if self.pseudo_dialog:
                self.pseudo_dialog.close()
            event.accept()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_font(app)
    window = MainWindow()
    window.show()
    if '--pseudo-depth' in sys.argv:
        window.open_pseudo_depth()
    elif len(sys.argv) > 1:
        try:
            window.import_bundle(sys.argv[1])
        except Exception as exc:
            window.report_error(str(exc))
    else:
        window.generate_simulation()
        QtCore.QTimer.singleShot(0, window.start_measurement)
    sys.exit(app.exec_())
