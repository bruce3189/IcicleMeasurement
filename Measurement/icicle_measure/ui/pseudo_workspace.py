"""Editable pseudo-depth workspace and its original background task lifecycle."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..pseudo_depth import (ROOT, DEFAULT_RGB_DIR, DEFAULT_LABEL_DIR, SAMPLE_STEMS,
    PseudoConfig, compose_scene, scene_from_inference, export_project, load_project, fixed_depth_rgb, depth_colorbar)
from ..pseudo_models import run_inference
from ..pixel_inspector import inspect_pixel, pixel_text
from ..render import draw_selected_distance, distance_summary
from .pseudo_preview import scene_preview, EditView


class PseudoTask(QtCore.QObject):
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()

    def __init__(self, function, stop):
        super().__init__()
        self.function, self.stop = function, stop

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.completed.emit(self.function(self.progress.emit, self.stop.is_set))
        except InterruptedError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(type(exc).__name__ + ': ' + str(exc))
        finally:
            self.finished.emit()


class PseudoDepthPanel(QtWidgets.QWidget):
    measurement_requested = QtCore.pyqtSignal(object, object, object)
    scene_ready = QtCore.pyqtSignal(object, object, object)
    measurement_invalidated = QtCore.pyqtSignal(str)
    target_selected = QtCore.pyqtSignal(str)
    task_stopped = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('RGB → 伪深度 · D435i Z16 仿真 / 非实测')
        self.resize(1380, 900)
        if self.screen() is not None:
            available = self.screen().availableGeometry()
            self.resize(min(1380, available.width() - 40), min(900, available.height() - 70))
        self.setStyleSheet('QPushButton{padding:4px 8px;} QComboBox{padding:4px;} '
                          'QTabBar::tab{padding:5px 8px;} QPlainTextEdit{padding:4px;}')
        self.result = None
        self.thread = self.worker = None
        self.stop = threading.Event()
        self.close_pending = False
        self.prompts = {}
        self.dirty = False
        self._syncing = False
        self.sample_roots = []
        self.last_root = None
        self.hover_model_depth = None
        self.measurement = None
        self.measurement_note = '待计算'
        self._job_changes_input = self._job_committed = False
        self._source_pending = False
        self._source_previous_dirty = False
        self._job_restore_source = False
        layout = QtWidgets.QVBoxLayout(self)
        badge = QtWidgets.QLabel('模型伪深度 / 人工合成 · 1280×720 · Z16 · 0.001 m/单位 · 0=无效\n'
            '距离先验 0.5–2 m；长度常见 18–22 cm，极限 17–25 cm。虚拟内参非标定，生成结果不能验证真实精度。')
        badge.setStyleSheet('background:#fff0c6;color:#75501c;padding:9px;')
        badge.setWordWrap(True)
        layout.addWidget(badge)
        files = QtWidgets.QGridLayout()
        self.rgb_path = QtWidgets.QLineEdit(str(DEFAULT_RGB_DIR / (SAMPLE_STEMS[0] + '.jpg')))
        self.label_path = QtWidgets.QLineEdit(str(DEFAULT_LABEL_DIR / (SAMPLE_STEMS[0] + '.txt')))
        self.rgb_browse = QtWidgets.QPushButton('原始 RGB…'); self.rgb_browse.clicked.connect(self.choose_rgb)
        self.labels_browse = QtWidgets.QPushButton('检测标签…'); self.labels_browse.clicked.connect(self.choose_labels)
        files.addWidget(self.rgb_browse, 0, 0); files.addWidget(self.rgb_path, 0, 1)
        files.addWidget(self.labels_browse, 1, 0); files.addWidget(self.label_path, 1, 1)
        layout.addLayout(files)
        commands = QtWidgets.QHBoxLayout()
        self.samples = QtWidgets.QComboBox()
        for stem in SAMPLE_STEMS:
            self.samples.addItem(stem)
        self.samples.currentTextChanged.connect(self.choose_sample)
        commands.addWidget(self.samples)
        self.generate_button = self.button(commands, '生成所选图片', self.generate)
        self.six_button = self.button(commands, '生成六张样例', self.generate_six)
        self.open_button = self.button(commands, '打开可编辑项目', self.open_project)
        self.save_button = self.button(commands, '保存项目 / Z16', self.save_project)
        self.cancel_button = self.button(commands, '取消', self.cancel)
        self.cancel_button.setEnabled(False)
        layout.addLayout(commands)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget(); left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        views = QtWidgets.QHBoxLayout()
        self.view_kind = QtWidgets.QComboBox()
        for title, data in [('RGB 与原检测', 'rgb'), ('实例掩膜 / 边界', 'masks'),
                            ('模型相对深度（非米）', 'relative'), ('原始模型映射（未修正）', 'model'), ('场景背景（含人工平面）', 'background'),
                            ('完整假想表面', 'complete'), ('带误差观测深度', 'observed'), ('不确定 / 缺陷来源', 'uncertainty')]:
            self.view_kind.addItem(title, data)
        self.view_kind.setCurrentIndex(1)
        self.view_kind.currentIndexChanged.connect(self.view_changed)
        views.addWidget(self.view_kind)
        self.variant = QtWidgets.QComboBox()
        self.variant.addItem('送测：带误差观测', 'observed'); self.variant.addItem('送测：完整表面', 'complete')
        self.variant.currentIndexChanged.connect(self.variant_changed)
        views.addWidget(self.variant)
        self.measure_button = self.button(views, '查看测距分析 / 点云', self.send_measurement)
        left_layout.addLayout(views)
        self.view = EditView()
        self.view.stroke.connect(self.paint_mask)
        self.view.stroke_finished.connect(self.finish_stroke)
        self.view.point_added.connect(self.add_prompt)
        self.view.pixel_hovered.connect(self.show_pixel_info)
        left_layout.addWidget(self.view, 1)
        self.colorbar = QtWidgets.QLabel()
        self.colorbar.setAlignment(QtCore.Qt.AlignCenter)
        self.colorbar.hide()
        left_layout.addWidget(self.colorbar)
        self.legend = QtWidgets.QLabel(); self.legend.setWordWrap(True)
        left_layout.addWidget(self.legend)
        self.hover_readout = QtWidgets.QLabel('将鼠标移到图像上查看像素深度；右侧“像素详情”提供完整读数。')
        self.hover_readout.setTextFormat(QtCore.Qt.PlainText)
        self.hover_readout.setWordWrap(True)
        self.hover_readout.setStyleSheet('background:#edf4fa;color:#203b50;padding:5px;font-size:12px;')
        left_layout.addWidget(self.hover_readout)
        self.distance_note = QtWidgets.QLabel('端点欧氏距离：待计算')
        self.distance_note.setWordWrap(True)
        self.distance_note.setStyleSheet('font-weight:600;color:#245c9b;padding:4px;')
        distance_row = QtWidgets.QHBoxLayout(); distance_row.addWidget(self.distance_note, 1)
        self.distance_check = QtWidgets.QCheckBox('显示当前目标距离'); self.distance_check.setChecked(True)
        self.distance_check.toggled.connect(self.refresh); distance_row.addWidget(self.distance_check)
        left_layout.addLayout(distance_row)
        split.addWidget(left)
        right = QtWidgets.QWidget(); right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        tabs = QtWidgets.QTabWidget()
        editing = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(editing)
        self.object_choice = QtWidgets.QComboBox(); self.object_choice.currentIndexChanged.connect(self.select_object)
        form.addRow('目标实例', self.object_choice)
        self.edit_mode = QtWidgets.QComboBox()
        for title, mode in [('查看 / 拖动', 'inspect'), ('正提示点', 'positive'), ('负提示点', 'negative'),
                            ('画笔增加掩膜', 'add'), ('画笔擦除掩膜', 'erase'), ('背景深度画笔', 'background'), ('圈选背景平面', 'plane')]:
            self.edit_mode.addItem(title, mode)
        self.edit_mode.currentIndexChanged.connect(lambda: self.view.set_mode(self.edit_mode.currentData()))
        form.addRow('编辑工具', self.edit_mode)
        self.brush = QtWidgets.QSpinBox(); self.brush.setRange(1, 80); self.brush.setValue(4)
        form.addRow('画笔直径 (px)', self.brush)
        self.refine_button = QtWidgets.QPushButton('按正负点重新分割当前目标')
        self.refine_button.clicked.connect(self.refine)
        form.addRow(self.refine_button)
        self.reset_mask_button = QtWidgets.QPushButton('恢复当前目标初始掩膜')
        self.reset_mask_button.clicked.connect(self.reset_mask)
        form.addRow(self.reset_mask_button)
        self.exclude_button = QtWidgets.QPushButton('忽略重复 / 误检（可恢复）')
        self.exclude_button.clicked.connect(self.exclude_object); form.addRow(self.exclude_button)
        self.overlap_button = QtWidgets.QPushButton('重叠像素归给当前目标')
        self.overlap_button.clicked.connect(self.assign_overlap); form.addRow(self.overlap_button)
        self.auto_base = QtWidgets.QCheckBox('使用自动基线'); self.auto_base.setChecked(True)
        form.addRow(self.auto_base)
        self.base = self.spin(.01, 20., 1.25, 3, .01); form.addRow('目标整体深度 (m)', self.base)
        self.slope = self.spin(-2., 2., 0., 3, .005); form.addRow('轴向首尾深度差 (m)', self.slope)
        self.background_delta = self.spin(-5., 5., .05, 3, .01); form.addRow('背景画笔偏移 (m)', self.background_delta)
        self.plane_points = []
        self.plane_depths = [self.spin(.05, 20., z, 3, .05) for z in (1.8, 2., 1.8)]
        for label, control in zip(('平面左上深度 (m)', '平面右上深度 (m)', '平面左下深度 (m)'), self.plane_depths):
            form.addRow(label, control)
        plane_note = QtWidgets.QLabel('沿墙面边界依次点击 ≥3 点，再确认。三个深度为圈选区域包围框的人工假设锚值；排除冰凌掩膜。')
        plane_note.setWordWrap(True); form.addRow(plane_note)
        self.plane_button = QtWidgets.QPushButton('确认圈选平面'); self.plane_button.clicked.connect(self.commit_plane)
        form.addRow(self.plane_button)
        self.undo_plane_button = QtWidgets.QPushButton('撤销圈选 / 最后一个平面'); self.undo_plane_button.clicked.connect(self.undo_plane)
        form.addRow(self.undo_plane_button)
        self.apply_button = QtWidgets.QPushButton('应用修正 / 重算观测缺陷')
        self.apply_button.clicked.connect(self.apply_edits)
        self.object_info = QtWidgets.QPlainTextEdit(); self.object_info.setReadOnly(True); self.object_info.setMinimumHeight(110)
        form.addRow(self.object_info)
        def scroll_panel(widget):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True); scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(widget); scroll.setMinimumSize(0, 0)
            return scroll
        tabs.addTab(scroll_panel(editing), '目标与修正')
        settings = QtWidgets.QWidget(); setting_form = QtWidgets.QFormLayout(settings)
        self.settings = {}
        for key, title, low, high, value, decimals, step in [
            ('near_m', '背景映射近端 (m)', .05, 20, .5, 3, .1),
            ('far_m', '背景映射远端 (m)', .1, 40, 4, 3, .1),
            ('fallback_m', '不明确目标基线 (m)', .5, 2, 1.25, 3, .05),
            ('hole_fraction', '目标成片空洞比例', 0, 1, .1, 3, .01),
            ('return_fraction', '背景回波比例', 0, 1, .05, 3, .01),
            ('edge_fraction', '边界错误比例', 0, 1, .2, 3, .01),
            ('noise_sigma_m_at_1m', '1 m 处噪声 σ (m)', 0, .1, .001, 4, .001),
            ('depth_scale_m_per_unit', 'Z16 每单位米值', .0001, .01, .001, 4, .0001)]:
            box = self.spin(low, high, value, decimals, step)
            self.settings[key] = box; setting_form.addRow(title, box)
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 2 ** 31 - 1); self.seed.setValue(42)
        setting_form.addRow('随机种子', self.seed)
        self.intrinsics_label = QtWidgets.QLabel('虚拟内参：由名义彩色视场角构造，非 D435i 标定')
        self.intrinsics_label.setWordWrap(True); setting_form.addRow(self.intrinsics_label)
        self.intrinsics_button = QtWidgets.QPushButton('导入原 RGB 分辨率的内参 JSON')
        self.intrinsics_button.clicked.connect(self.choose_intrinsics); setting_form.addRow(self.intrinsics_button)
        self.assumed_button = QtWidgets.QPushButton('恢复假设内参'); self.assumed_button.clicked.connect(self.reset_intrinsics)
        setting_form.addRow(self.assumed_button)
        explanation = QtWidgets.QLabel('误差参数为工程测试假设，尚未用 D435i 实测标定。\n'
            '长度常见 18–22 cm、极限 17–25 cm，仅提示，不参与拟合。\n'
            '统一预览色标 0.5–4 m；黑色无效，色标之外截色而不改存储值。')
        explanation.setWordWrap(True); setting_form.addRow(explanation)
        tabs.addTab(scroll_panel(settings), '尺度与缺陷')
        self.pixel_details = QtWidgets.QPlainTextEdit()
        self.pixel_details.setReadOnly(True)
        self.pixel_details.setPlainText('将鼠标移到图像上，实时查看该像素的深度和来源。')
        tabs.addTab(self.pixel_details, '像素详情')
        self.panel = tabs
        right_layout.addWidget(tabs, 1)
        right_layout.addWidget(self.apply_button)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMinimumHeight(45); self.log.setMaximumHeight(80)
        right_layout.addWidget(self.log)
        split.addWidget(right); split.setSizes([950, 400])
        layout.addWidget(split, 1)
        self.status = QtWidgets.QLabel('先生成一张或打开已生成的六张样例。整批 400 张处理待样例确认后开放。')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.controls = [self.rgb_browse, self.labels_browse, self.rgb_path, self.label_path, self.samples,
                         self.generate_button, self.six_button, self.open_button, self.panel, self.apply_button]
        self.custom_intrinsics, self.custom_intrinsics_source = None, 'assumed_virtual_pinhole'
        for box in list(self.settings.values()) + [self.seed]:
            box.valueChanged.connect(self.mark_dirty)
        for box in [self.base, self.slope]:
            box.valueChanged.connect(self.object_parameter_changed)
        self.auto_base.toggled.connect(self.object_parameter_changed)
        self.rgb_path.textChanged.connect(self.mark_dirty)
        self.label_path.textChanged.connect(self.mark_dirty)
        self.set_busy(False)

    def mark_dirty(self, *_):
        if self.result is not None and not self._syncing:
            if self.sender() in (self.rgb_path, self.label_path) and not self._source_pending:
                self._source_previous_dirty = self.dirty
                self._source_pending = True
            self.invalidate_edits()
            self.status.setText('参数已修改；请点击“应用修正”后保存或送入测距。')

    def invalidate_edits(self):
        self.dirty = True
        self.clear_measurement('修正未应用，测距结果待更新')
        self.measurement_invalidated.emit(self.measurement_note)

    def selected_frame(self):
        if self.result is None:
            return None
        return self.result.frame if self.variant.currentData() == 'observed' else self.result.clean_frame

    def emit_scene(self):
        if self.result is not None and not self.dirty and not self.close_pending:
            self.scene_ready.emit(self.selected_frame(), self.result.scene.detections, self.result.scene.intrinsics)

    def variant_changed(self, *_):
        if self.result is not None and not self._syncing:
            self.clear_measurement('正在按新深度版本计算…')
            if not self.dirty:
                self.emit_scene()
        self.refresh()

    def view_changed(self, *_):
        kind = self.view_kind.currentData()
        if kind in ('complete', 'observed') and not self._syncing:
            self.variant.setCurrentIndex(self.variant.findData(kind))
        self.refresh()

    def clear_measurement(self, reason='待计算'):
        self.measurement = None
        self.measurement_note = reason
        if hasattr(self, 'distance_note'):
            self.refresh()

    def set_measurement_result(self, measurement):
        if self.dirty:
            return
        self.measurement = measurement
        self.measurement_note = '待计算'
        self.refresh()

    def object_parameter_changed(self, *_):
        self.base.setEnabled(not self.auto_base.isChecked())
        if self.result is not None and not self._syncing and self.object_choice.currentIndex() >= 0:
            did = self.result.scene.detections[self.object_choice.currentIndex()].id
            self.result.scene.edits['objects'][did] = {
                'base_m': None if self.auto_base.isChecked() else self.base.value(), 'slope_m': self.slope.value()}
            self.mark_dirty()

    def check_current_source(self):
        if self.result is not None and (Path(self.rgb_path.text()).resolve() != Path(self.result.scene.source_rgb).resolve()
                or Path(self.label_path.text()).resolve() != Path(self.result.scene.source_labels).resolve()):
            raise ValueError('RGB 或标签路径已更换，请先点击“生成所选图片”；不能将旧场景修正到新文件上。')

    def button(self, layout, title, callback):
        button = QtWidgets.QPushButton(title); button.clicked.connect(callback); layout.addWidget(button)
        return button

    @staticmethod
    def spin(low, high, value, decimals, step):
        box = QtWidgets.QDoubleSpinBox(); box.setRange(low, high); box.setDecimals(decimals)
        box.setSingleStep(step); box.setValue(value)
        return box

    def config(self):
        cfg = PseudoConfig(**{k: w.value() for k, w in self.settings.items()}, seed=self.seed.value(),
            intrinsics=self.custom_intrinsics, intrinsics_source=self.custom_intrinsics_source)
        cfg.validate()
        return cfg

    def choose_sample(self, stem):
        syncing = self._syncing
        self._syncing = True
        self.rgb_path.setText(str(DEFAULT_RGB_DIR / (stem + '.jpg')))
        self.label_path.setText(str(DEFAULT_LABEL_DIR / (stem + '.txt')))
        self._syncing = syncing
        generated = ROOT / 'outputs' / 'pseudo_depth_samples' / stem / 'pseudo_project.json'
        improved = ROOT / 'outputs' / 'pseudo_depth_refined' / stem / 'pseudo_project.json'
        if improved.exists():
            generated = improved
        if generated.exists() and self.thread is None:
            self.launch(lambda progress, cancel: {'result': load_project(generated), 'root': generated.parent}, '加载样例')

    def choose_rgb(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择未画框的原始 RGB', str(DEFAULT_RGB_DIR), 'Images (*.jpg *.jpeg *.png *.bmp)')
        if path:
            self.rgb_path.setText(path)
            candidate = DEFAULT_LABEL_DIR / (Path(path).stem + '.txt')
            if candidate.exists():
                self.label_path.setText(str(candidate))

    def choose_labels(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择对应的两点预测标签', str(DEFAULT_LABEL_DIR), 'Labels (*.txt *.json)')
        if path:
            self.label_path.setText(path)

    def choose_intrinsics(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '原始 RGB 的内参：fx,fy,cx,cy', '', 'JSON (*.json)')
        if path:
            try:
                from ..models import Intrinsics
                from ..io import read_json
                meta = read_json(path); values = meta.get('intrinsics', meta)
                Intrinsics(**values).validate()
                expected = meta.get('intrinsics_image_size')
                if expected is not None:
                    from ..io import read_image
                    image = read_image(self.rgb_path.text(), cv2.IMREAD_COLOR)
                    if list(expected) != [image.shape[1], image.shape[0]]:
                        raise ValueError('内参图像尺寸与当前原始 RGB 不符')
                self.custom_intrinsics, self.custom_intrinsics_source = values, 'user_supplied:' + str(path)
                self.intrinsics_label.setText('用户提供内参：' + str(path) + '（原 RGB 分辨率，未声明已去畸变）')
                self.invalidate_edits()
            except Exception as exc:
                self.error(str(exc))

    def reset_intrinsics(self):
        self.custom_intrinsics, self.custom_intrinsics_source = None, 'assumed_virtual_pinhole'
        self.intrinsics_label.setText('虚拟内参：由名义彩色视场角构造，非 D435i 标定')
        self.invalidate_edits()

    @staticmethod
    def build_one(rgb, labels, cfg, progress, cancel):
        from ..core import measure_frame
        from ..io import export_results
        begin = time.perf_counter()
        cache = run_inference(rgb, labels, cfg, progress=progress, cancel=cancel)
        scene = scene_from_inference(rgb, labels, cache, cfg)
        result = compose_scene(scene, cfg, cancel)
        if cancel():
            raise InterruptedError('用户取消')
        root = ROOT / 'outputs' / 'pseudo_depth_samples' / Path(rgb).stem
        export_project(root, result)
        outcomes = {}
        for name, frame in [('observed', result.frame), ('complete', result.clean_frame)]:
            measured = measure_frame(frame, scene.detections, scene.intrinsics, cancel=cancel)
            export_results(root / ('measurement_' + name), frame, scene.detections, measured)
            outcomes[name] = [{'id': obj.id, 'status': obj.status, 'reason': obj.reason,
                               'distance_m': obj.distance_m} for obj in measured.objects]
        if cancel():
            raise InterruptedError('用户取消')
        from ..io import write_json
        elapsed = time.perf_counter() - begin
        write_json(root / 'timing_and_measurement.json', {'seconds_including_io': elapsed,
            'model_seconds_uncached': scene.model_info.get('seconds'), 'measurement': outcomes,
            'is_synthetic': True, 'accuracy_validation': False})
        progress({'message': '%s 完成，%.1f 秒' % (Path(rgb).stem, elapsed)})
        return {'result': result, 'root': root, 'seconds': elapsed, 'measurement': outcomes}

    def generate(self):
        try:
            cfg = self.config(); rgb, labels = self.rgb_path.text(), self.label_path.text()
            if not Path(rgb).is_file() or not Path(labels).is_file():
                raise ValueError('请选择存在且相互对应的原 RGB 和结构化标签文件')
            self.launch(lambda progress, cancel: self.build_one(rgb, labels, cfg, progress, cancel), '生成当前图片')
        except Exception as exc:
            self.error(str(exc))

    def generate_six(self):
        try:
            cfg = self.config()
            def batch(progress, cancel):
                rows = []
                latest = None
                for stem in SAMPLE_STEMS:
                    if cancel():
                        raise InterruptedError('用户取消；已完成样例保留在输出目录')
                    try:
                        latest = self.build_one(DEFAULT_RGB_DIR / (stem + '.jpg'), DEFAULT_LABEL_DIR / (stem + '.txt'), cfg, progress, cancel)
                        rows.append({'image': stem, 'seconds': latest['seconds'], 'status': 'complete'})
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        rows.append({'image': stem, 'status': 'failed', 'reason': str(exc)})
                        progress({'message': stem + ' 失败：' + str(exc)})
                from ..io import write_json
                write_json(ROOT / 'outputs' / 'pseudo_depth_samples' / 'batch_report.json', {'samples': rows})
                if latest is None:
                    raise RuntimeError('六张样例均未生成成功，请查看日志')
                latest['batch'] = rows
                return latest
            self.launch(batch, '依次生成六张样例')
        except Exception as exc:
            self.error(str(exc))

    def refine(self):
        if not self.result or self.object_choice.currentIndex() < 0:
            return
        try:
            self.check_current_source()
            scene, cfg = copy.deepcopy(self.result.scene), self.config()
        except Exception as exc:
            self.error(str(exc)); return
        did = scene.detections[self.object_choice.currentIndex()].id
        prompts = copy.deepcopy(self.prompts)
        if not prompts.get(did):
            self.error('请先在当前目标上添加正点或负点；原关键点不会自动作为分割提示。'); return
        def job(progress, cancel):
            cache = run_inference(scene.source_rgb, scene.source_labels, cfg, prompts, progress, cancel,
                                   reuse=scene.model_info.get('cache_file'), changed_ids=[did])
            inferred = scene_from_inference(scene.source_rgb, scene.source_labels, cache, cfg)
            index = next(i for i, d in enumerate(scene.detections) if d.id == did)
            scene.masks[index] = inferred.masks[index]
            scene.model_info = inferred.model_info
            scene.intrinsics = inferred.intrinsics
            scene.edits['actions'].append({'kind': 'sam_prompts', 'id': did, 'points': prompts[did]})
            return {'result': compose_scene(scene, cfg, cancel), 'root': self.last_root}
        self.launch(job, '重新分割所选目标')

    def apply_edits(self):
        if self.result is None:
            return
        try:
            self.check_current_source()
            cfg = self.config(); scene = copy.deepcopy(self.result.scene)
            index = self.object_choice.currentIndex()
            if index >= 0:
                did = scene.detections[index].id
                edit = {'base_m': None if self.auto_base.isChecked() else self.base.value(), 'slope_m': self.slope.value()}
                scene.edits['objects'][did] = edit
                scene.edits['actions'].append({'kind': 'object_depth', 'id': did, **edit})
            from ..pseudo_depth import virtual_intrinsics
            scene.intrinsics = virtual_intrinsics(cfg, scene.transform)
            self.launch(lambda progress, cancel: {'result': compose_scene(scene, cfg, cancel), 'root': self.last_root}, '应用修正')
        except Exception as exc:
            self.error(str(exc))

    def paint_mask(self, stroke):
        if self.result is None or self.object_choice.currentIndex() < 0 or self.thread:
            return
        scene = self.result.scene
        a, b, mode = stroke['start'], stroke['end'], stroke['mode']
        region = np.zeros(scene.valid.shape, np.uint8)
        cv2.line(region, a, b, 1, self.brush.value(), cv2.LINE_8)
        region = region.astype(bool) & scene.valid
        index = self.object_choice.currentIndex()
        if mode == 'background':
            region &= ~scene.masks.any(axis=0)
            scene.background_offset[region] = self.background_delta.value()
        elif mode in ('add', 'erase'):
            scene.masks[index][region] = mode == 'add'
        scene.edits['actions'].append({'kind': 'brush', 'id': scene.detections[index].id, 'mode': mode,
            'start': a, 'end': b, 'diameter_px': self.brush.value(),
            'offset_m': self.background_delta.value() if mode == 'background' else None})
        self.invalidate_edits()
        self.view_kind.setCurrentIndex(self.view_kind.findData('masks'))
        self.refresh()

    def finish_stroke(self):
        self.status.setText('掩膜或背景已修改；点击“应用修正”重建深度，随后保存或送入测距。')

    def add_prompt(self, point):
        if self.result is not None and point.get('mode') == 'plane' and not self.thread:
            self.plane_points.append((point['x'], point['y']))
            self.refresh()
            self.status.setText('已圈选 %d 点；设置三个假设深度后确认平面。' % len(self.plane_points))
            return
        if self.result is None or self.object_choice.currentIndex() < 0:
            return
        did = self.result.scene.detections[self.object_choice.currentIndex()].id
        self.prompts.setdefault(did, []).append(point)
        self.result.scene.edits['prompts'] = copy.deepcopy(self.prompts)
        self.refresh()
        self.status.setText('提示点已记录；点击“按正负点重新分割当前目标”运行 MobileSAM。')

    def reset_mask(self):
        if self.result is not None and self.object_choice.currentIndex() >= 0:
            s = self.result.scene; i = self.object_choice.currentIndex()
            s.masks[i] = s.original_masks[i]
            self.prompts.pop(s.detections[i].id, None)
            s.edits['actions'].append({'kind': 'reset_mask', 'id': s.detections[i].id})
            self.invalidate_edits(); self.refresh(); self.finish_stroke()

    def exclude_object(self):
        if self.thread or self.result is None or self.object_choice.currentIndex() < 0:
            return
        s = self.result.scene; i = self.object_choice.currentIndex()
        s.masks[i][:] = False
        s.edits['actions'].append({'kind': 'manual_exclude_detection', 'id': s.detections[i].id,
                                  'reason': '用户标记重复或误检；保留原检测记录，取消其表面'})
        self.invalidate_edits(); self.refresh(); self.finish_stroke()

    def assign_overlap(self):
        if self.thread or self.result is None or self.object_choice.currentIndex() < 0:
            return
        s = self.result.scene; index = self.object_choice.currentIndex()
        overlap = s.masks[index] & (s.masks.sum(axis=0) > 1)
        changed = []
        for i, mask in enumerate(s.masks):
            if i != index and np.any(mask & overlap):
                mask[overlap] = False; changed.append(s.detections[i].id)
        s.edits['actions'].append({'kind': 'manual_overlap_owner', 'id': s.detections[index].id,
                                  'affected_ids': changed, 'pixels': int(overlap.sum())})
        self.invalidate_edits(); self.refresh(); self.finish_stroke()

    def commit_plane(self):
        if self.result is None or self.thread:
            return
        if len(self.plane_points) < 3:
            self.error('请用圈选背景平面工具沿区域边界点击至少三个点'); return
        plane = {'polygon': list(self.plane_points), 'depths_m': [w.value() for w in self.plane_depths],
                 'source': 'manual_scene_hypothesis'}
        from ..pseudo_geometry import apply_background_planes
        try:
            s = self.result.scene
            apply_background_planes(self.result.background_m, s.valid, s.masks.any(axis=0), [plane])
        except ValueError as exc:
            self.error(str(exc)); return
        s.edits.setdefault('background_planes', []).append(plane)
        self.plane_points.clear(); self.invalidate_edits()
        self.apply_edits()

    def undo_plane(self):
        if self.thread or self.result is None:
            return
        if self.plane_points:
            self.plane_points.clear(); self.refresh(); return
        planes = self.result.scene.edits.get('background_planes', [])
        if planes:
            planes.pop(); self.invalidate_edits(); self.apply_edits()

    def select_object(self):
        if self.result is None:
            return
        i = self.object_choice.currentIndex()
        if i < 0:
            return
        d = self.result.scene.detections[i]
        syncing = self._syncing
        self._syncing = True
        edit = self.result.scene.edits['objects'].get(d.id, {})
        info = self.result.report['objects'][i]
        self.auto_base.setChecked(edit.get('base_m') is None)
        self.base.setEnabled(not self.auto_base.isChecked())
        self.base.setValue(edit.get('base_m') or info.get('base_m', self.result.config.fallback_m))
        self.slope.setValue(edit.get('slope_m', 0.))
        sources = {'manual': '人工指定', 'model_candidate': '模型候选，经假设尺度映射',
                   'assumed_distance_prior': '距离先验假设', 'assumed_1_25m': '距离先验假设'}
        lines = ['状态：' + {'assumed_surface': '自动假想表面（非实测）', 'manual_surface': '人工修正表面',
                             'needs_mask_review': '掩膜待修正'}.get(info['status'], info['status'])]
        if 'base_m' in info:
            lines += ['基线深度：%.3f m（%s）' % (info['base_m'], sources.get(info['base_source'], info['base_source'])),
                      '掩膜样本：%d px' % info['mask_pixels'],
                      '轮廓弦长：%.1f cm；常见 %.0f–%.0f cm，极限 %.0f–%.0f cm（仅提示）' % (
                          info['silhouette_chord_m'] * 100, self.result.config.length_nominal_min_m * 100,
                          self.result.config.length_nominal_max_m * 100,
                          self.result.config.length_min_m * 100, self.result.config.length_max_m * 100),
                      '模型前景支持：' + ('有，仍非实测依据' if info['model_foreground_support'] else '不足，已使用显式假设')]
        lines += ['提示：' + text for text in info.get('warnings', [])]
        if info.get('reason'):
            lines.append(info['reason'])
        self.object_info.setPlainText('\n'.join(lines))
        self._syncing = syncing
        self.refresh()
        if not syncing:
            self.target_selected.emit(d.id)

    def refresh(self, fit=False):
        if self.result is None:
            return
        kind = self.view_kind.currentData()
        self.hover_model_depth = None
        if kind == 'model':
            from ..pseudo_depth import _map_relative
            self.hover_model_depth, _ = _map_relative(self.result.scene.relative, self.result.scene.valid, self.result.config)
        image = scene_preview(self.result, kind, self.object_choice.currentIndex())
        selected_id = self.object_choice.currentData()
        measured = next((obj for obj in self.measurement.objects if obj.id == selected_id), None) if self.measurement else None
        self.distance_note.setText(distance_summary(measured, True, self.variant.currentData(), self.measurement_note))
        if self.distance_check.isChecked() and kind in ('rgb','masks','complete','observed') and selected_id is not None:
            detection = next((d for d in self.result.scene.detections if d.id == selected_id), None)
            if detection is not None:
                draw_selected_distance(image, detection, measured, 'stale' if self.dirty else 'pending')
        for plane in self.result.scene.edits.get('background_planes', []):
            cv2.polylines(image, [np.asarray(plane['polygon'], np.int32)], True, (255, 200, 70), 1)
        if self.plane_points:
            points = np.asarray(self.plane_points, np.int32)
            cv2.polylines(image, [points], False, (255, 220, 60), 2)
            for point in points:
                cv2.circle(image, tuple(point), 3, (255, 220, 60), -1)
        if kind in ('rgb', 'masks') and self.object_choice.currentIndex() >= 0:
            did = self.result.scene.detections[self.object_choice.currentIndex()].id
            for p in self.prompts.get(did, []):
                color = (65, 255, 100) if p['positive'] else (255, 70, 65)
                cv2.drawMarker(image, (p['x'], p['y']), color, cv2.MARKER_CROSS, 11, 2)
        self.view.set_image(image, fit=bool(fit))
        position = self.view.viewport().mapFromGlobal(QtGui.QCursor.pos())
        self.show_pixel_info(self.view.pixel_at(position) if self.view.viewport().rect().contains(position) else None)
        if kind in ('observed', 'complete', 'background', 'model'):
            cfg = self.result.config
            bar = depth_colorbar(cfg.color_min_m, cfg.color_max_m)
            self.colorbar.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(bar.data, bar.shape[1], bar.shape[0],
                bar.strides[0], QtGui.QImage.Format_RGB888).copy()))
            self.colorbar.show()
            self.legend.setText('统一深度色标：%.2f → %.2f m（近：蓝，远：红）；黑色无效；超出色标范围仅截色。'
                ' 数值为人工设定尺度，非相机实测。' % (cfg.color_min_m, cfg.color_max_m))
        elif kind == 'uncertainty':
            self.colorbar.hide()
            self.legend.setText('蓝：假想目标表面；橙：掩膜待修；紫：实例重叠；白：空洞；黄：背景回波；红：边界错误。来源分类不是正确概率。')
        elif kind == 'relative':
            self.colorbar.hide()
            self.legend.setText('模型原始相对深度：较大值更近，非米；显示使用本图记录的 P2/P98 归一化，不能据此比较跨图绝对距离。')
        else:
            self.colorbar.hide()
            self.legend.setText('原预测关键点：红=尖端，蓝=根部；绿边为所选目标。滚轮缩放，查看模式拖动，双击适配。画笔以输出像素为单位。')

    def show_pixel_info(self, pixel):
        if pixel is None or self.result is None:
            self.hover_readout.setText('将鼠标移到图像上查看像素深度；右侧“像素详情”提供完整读数。')
            self.pixel_details.setPlainText('鼠标不在图像有效显示范围内。')
            return
        kind = self.view_kind.currentData()
        layer = kind if kind in ('relative','model','background','complete','observed') else self.variant.currentData()
        info = inspect_pixel(self.result, *pixel, layer, model_depth=self.hover_model_depth)
        if info is None:
            self.show_pixel_info(None); return
        compact, detail = pixel_text(info)
        if self.dirty:
            compact += '\n修正尚未应用：读数对应上次生成的深度。'
            detail = '修正尚未应用：以下为上次生成深度的读数。\n\n' + detail
        self.hover_readout.setText(compact)
        # Updating text only for a changed pixel avoids needless layout on stationary cursors.
        if self.pixel_details.toPlainText() != detail:
            self.pixel_details.setPlainText(detail)

    def send_measurement(self):
        if self.result is None:
            return
        if self.dirty:
            self.error('请先点击“应用修正”，以免将旧深度送入测距。'); return
        result = self.result
        frame = result.frame if self.variant.currentData() == 'observed' else result.clean_frame
        self.measurement_requested.emit(frame, result.scene.detections, result.scene.intrinsics)
        self.status.setText('已切换测距分析工作页；结果仍为伪深度估计。')

    def save_project(self):
        if self.result is None:
            return
        if self.dirty:
            self.error('请先应用修正，再保存与深度一致的项目。'); return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, '保存项目（同名文件会覆盖）', str(self.last_root or ROOT / 'outputs'))
        if directory:
            result = copy.deepcopy(self.result)
            self.launch(lambda progress, cancel: {'saved_root': Path(export_project(directory, result)).parent}, '保存项目', changes_input=False)

    def open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择 pseudo_project.json', str(ROOT / 'outputs' / 'pseudo_depth_samples'), 'JSON (*.json)')
        if path:
            self.launch(lambda progress, cancel: {'result': load_project(path), 'root': Path(path).parent}, '加载可编辑项目')

    def launch(self, function, message, changes_input=True):
        if self.thread is not None:
            return
        self.stop = threading.Event()
        self._job_changes_input, self._job_committed = changes_input, False
        self._job_restore_source = bool(changes_input and self.result is not None and
            (self.rgb_path.text() != self.result.scene.source_rgb or self.label_path.text() != self.result.scene.source_labels))
        self._job_prior_dirty = self._source_previous_dirty if self._source_pending else self.dirty
        if changes_input:
            self.clear_measurement('正在更新输入，测距结果待更新')
            self.measurement_invalidated.emit(self.measurement_note)
        self.set_busy(True); self.status.setText(message)
        self.thread = QtCore.QThread(self)
        self.worker = PseudoTask(function, self.stop)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.completed.connect(self.receive)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.stopped)
        self.thread.start()

    def progress(self, value):
        message = value.get('message', str(value))
        self.status.setText(message); self.log.appendPlainText(message)

    def receive(self, payload):
        if 'saved_root' in payload:
            self.last_root = payload['saved_root']
            self.status.setText('项目已保存；当前测距与显示状态保留。')
            return
        self._job_committed = True
        self._source_pending = False
        self.measurement = None
        self.measurement_note = '正在自动测距…'
        self.plane_points.clear()
        selected_id = self.object_choice.currentData()
        self.result, self.last_root = payload['result'], payload.get('root')
        self._syncing = True
        self.dirty = False
        cfg = self.result.config
        for key, box in self.settings.items():
            box.setValue(getattr(cfg, key))
        self.seed.setValue(cfg.seed)
        self.custom_intrinsics, self.custom_intrinsics_source = cfg.intrinsics, cfg.intrinsics_source
        self.intrinsics_label.setText('用户提供内参：' + cfg.intrinsics_source if cfg.intrinsics
            else '虚拟内参：由名义彩色视场角构造，非 D435i 标定')
        s = self.result.scene
        sample_index = self.samples.findText(Path(s.source_rgb).stem)
        self.samples.blockSignals(True); self.samples.setCurrentIndex(sample_index); self.samples.blockSignals(False)
        self.rgb_path.setText(s.source_rgb); self.label_path.setText(s.source_labels)
        self.prompts = copy.deepcopy(s.edits.get('prompts', {}))
        for item in s.edits.get('actions', []):
            if item['kind'] == 'sam_prompts':
                self.prompts[item['id']] = item['points']
        self.object_choice.blockSignals(True); self.object_choice.clear()
        for d, info in zip(s.detections, self.result.report['objects']):
            status = {'assumed_surface': '假想表面', 'manual_surface': '人工修正',
                      'needs_mask_review': '掩膜待修'}.get(info['status'], info['status'])
            self.object_choice.addItem('#' + d.id + ' · ' + status, d.id)
        index = self.object_choice.findData(selected_id)
        self.object_choice.setCurrentIndex(max(0, index)); self.object_choice.blockSignals(False)
        self.select_object(); self.refresh(fit=True)
        self._syncing = False
        self.status.setText('完成：%d 个目标，%d 个目标像素；空洞 %d、背景回波 %d、重叠无效 %d。'
            '正在自动测距，可切换“测距分析”查看。' % (len(s.detections), self.result.report['target_pixels'],
            self.result.report['hole_pixels'], self.result.report['return_pixels'], self.result.report['overlap_pixels']))
        self.set_busy(self.thread is not None)
        self.emit_scene()

    def stopped(self):
        thread = self.thread
        self.thread = self.worker = None
        if thread:
            thread.deleteLater()
        self.set_busy(False)
        if self._job_restore_source and not self._job_committed and self.result is not None:
            self._syncing = True
            self.rgb_path.setText(self.result.scene.source_rgb)
            self.label_path.setText(self.result.scene.source_labels)
            self.samples.blockSignals(True)
            self.samples.setCurrentIndex(self.samples.findText(Path(self.result.scene.source_rgb).stem))
            self.samples.blockSignals(False)
            self._syncing = False
            self.dirty = self._job_prior_dirty
            self._source_pending = False
        if self._job_changes_input and not self._job_committed and self.result is not None and not self.dirty and not self.close_pending:
            self.status.setText(self.status.text() + '；保留上次成功输入，重新显示其测距结果。')
            self.emit_scene()
        self.task_stopped.emit()
        if self.close_pending:
            self.close()

    def set_busy(self, busy):
        for control in self.controls:
            control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.view.setEnabled(not busy)
        self.save_button.setEnabled(not busy and self.result is not None)
        self.measure_button.setEnabled(not busy and self.result is not None)

    def cancel(self):
        self.stop.set(); self.status.setText('正在取消模型任务或等待当前合成步骤结束…')

    def error(self, message):
        self.status.setText(message); self.log.appendPlainText(message)

    def closeEvent(self, event):
        if self.thread is not None:
            self.close_pending = True; self.cancel(); event.ignore()
        else:
            event.accept()


class PseudoDepthDialog(PseudoDepthPanel):
    """Compatibility wrapper for existing standalone utilities; the app embeds Panel."""
    finished = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Window, True)

    def closeEvent(self, event):
        super().closeEvent(event)
        if event.isAccepted():
            self.finished.emit(0)
