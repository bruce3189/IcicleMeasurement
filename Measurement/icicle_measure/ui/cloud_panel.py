"""Controls and detail readout around the stable Qt point-cloud canvas."""
import numpy as np
from PyQt5 import QtCore, QtWidgets
from .cloud_canvas import DepthTerrainCanvas


class DepthTerrainPanel(QtWidgets.QWidget):
    expand_requested = QtCore.pyqtSignal(bool)

    def __init__(self, allow_large_view=True):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self.distance_note = QtWidgets.QLabel('端点估计欧氏距离：待计算')
        self.distance_note.setWordWrap(True)
        self.distance_note.setStyleSheet('font-size:12px;color:#245c9b;')
        header = QtWidgets.QHBoxLayout(); header.addWidget(self.distance_note, 1)
        self.settings_toggle = QtWidgets.QToolButton(); self.settings_toggle.setText('点云设置')
        self.settings_toggle.setCheckable(True); self.settings_toggle.setChecked(True)
        header.addWidget(self.settings_toggle)
        self.expand_button = QtWidgets.QToolButton(); self.expand_button.setText('展开点云')
        self.expand_button.setCheckable(True)
        self.expand_button.toggled.connect(self.expand_requested.emit)
        self.expand_button.setVisible(allow_large_view)
        header.addWidget(self.expand_button); layout.addLayout(header)
        settings_widget = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0); settings_layout.setSpacing(3)
        self.settings_scroll = QtWidgets.QScrollArea()
        self.settings_scroll.setWidgetResizable(True); self.settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.settings_scroll.setWidget(settings_widget); self.settings_scroll.setMaximumHeight(180)
        self.settings_scroll.setMinimumSize(0, 0)
        self.settings_toggle.toggled.connect(self.settings_scroll.setVisible)
        layout.addWidget(self.settings_scroll)
        self.view_options = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(self.view_options)
        view_layout.setContentsMargins(0, 0, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        self.canvas = DepthTerrainCanvas()
        self.canvas.navigation_changed.connect(self.update_navigation)
        navigation = QtWidgets.QHBoxLayout()
        self.focus_button = QtWidgets.QToolButton(); self.focus_button.setText('聚焦样本 F')
        self.focus_button.setToolTip('先单击样本，再聚焦并围绕它旋转；不移动端点坐标原点')
        self.focus_button.clicked.connect(self.focus_sample)
        navigation.addWidget(self.focus_button)
        for label, callback in [('适配点云',self.canvas.fit_view),('返回端点',self.canvas.return_to_endpoint),('复位视角',self.canvas.reset_view)]:
            button = QtWidgets.QToolButton(); button.setText(label); button.clicked.connect(callback); navigation.addWidget(button)
        self.navigation_info = QtWidgets.QLabel('WASD 平移 · 平滑滚轮 · Shift 精调')
        self.navigation_info.setWordWrap(True)
        self.navigation_info.setStyleSheet('font-size:11px;color:#53677c;')
        navigation.addWidget(self.navigation_info,1)
        layout.addLayout(navigation)
        controls.addWidget(QtWidgets.QLabel('坐标轴：'))
        for axis in ('x', 'y', 'z'):
            checkbox = QtWidgets.QCheckBox(axis.upper())
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda checked, axis=axis: self.canvas.set_axis_visible(axis, checked))
            controls.addWidget(checkbox)
        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel('旋转：'))
        for axis, label in (("x", "X"), ("y", "Y"), ("z", "Z")):
            for degrees, suffix in ((-15, "−"), (15, "+")):
                button = QtWidgets.QToolButton()
                button.setText(label + suffix)
                button.setToolTip('绕 %s 轴旋转 %s15°' % (label, '负向 ' if degrees < 0 else '正向 '))
                button.clicked.connect(lambda checked=False, axis=axis, degrees=degrees:
                                       self.canvas.rotate_axis(axis, degrees))
                controls.addWidget(button)
        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setText('复位视角')
        self.reset_button.setToolTip('恢复固定默认视角、缩放和位置')
        self.reset_button.clicked.connect(self.canvas.reset_view)
        controls.addWidget(self.reset_button)
        controls.addStretch(1)
        view_layout.addLayout(controls)
        view_controls = QtWidgets.QHBoxLayout()
        view_controls.addWidget(QtWidgets.QLabel('固定观察方向：'))
        for label, eye, description in (
                ('光轴 −Z', (0., 0., -1.), '沿相机光轴观察点云'),
                ('+Z', (0., 0., 1.), '从 +Z 一侧观察点云'),
                ('+X', (1., 0., 0.), '从 +X 一侧观察点云'),
                ('−X', (-1., 0., 0.), '从 −X 一侧观察点云'),
                ('+Y', (0., 1., 0.), '从 +Y 一侧观察点云'),
                ('−Y', (0., -1., 0.), '从 −Y 一侧观察点云')):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setToolTip(description)
            button.clicked.connect(lambda checked=False, eye=eye:
                                   self.canvas.set_view_from_direction(eye))
            view_controls.addWidget(button)
        view_controls.addStretch(1)
        view_layout.addLayout(view_controls)
        mode_controls = QtWidgets.QHBoxLayout()
        mode_controls.addWidget(QtWidgets.QLabel('投影：'))
        self.projection_combo = QtWidgets.QComboBox()
        self.projection_combo.addItem('透视（虚拟针孔）', 'perspective')
        self.projection_combo.addItem('正交（平行投影）', 'orthographic')
        self.projection_combo.setToolTip('虚拟针孔透视：近处点放大，绘制顺序随观察视角变化')
        self.projection_combo.currentIndexChanged.connect(
            lambda index: self.canvas.set_projection_mode(self.projection_combo.itemData(index)))
        mode_controls.addWidget(self.projection_combo)
        mode_controls.addSpacing(12)
        mode_controls.addWidget(QtWidgets.QLabel('深度：'))
        self.depth_mode_combo = QtWidgets.QComboBox()
        self.depth_mode_combo.addItem('真实等比（米制）', 'true')
        self.depth_mode_combo.addItem('夸张 Z×4（仅观察）', 'exaggerated')
        self.depth_mode_combo.setToolTip('夸张模式只放大 Z 轴，形状已失真，坐标不再等比')
        self.depth_mode_combo.currentIndexChanged.connect(
            lambda index: self.canvas.set_depth_display_mode(self.depth_mode_combo.itemData(index)))
        mode_controls.addWidget(self.depth_mode_combo)
        mode_controls.addSpacing(12)
        mode_controls.addWidget(QtWidgets.QLabel('范围：'))
        self.focus_combo = QtWidgets.QComboBox()
        self.focus_combo.addItem('搜索窗全部点', 'all')
        self.focus_combo.addItem('像素圆邻域（保留前后景）', 'pixel')
        self.focus_combo.addItem('三维球邻域（可能隐藏背景）', 'local')
        self.focus_combo.setToolTip('像素邻域按原图距离筛选，不按深度剔除背景；三维邻域按真实 XYZ 距离筛选。空范围不会补入其它点。')
        self.focus_combo.setCurrentIndex(0)
        self.focus_combo.currentIndexChanged.connect(self.change_focus)
        mode_controls.addWidget(self.focus_combo)
        mode_controls.addStretch(1)
        settings_layout.addLayout(mode_controls)
        extras = QtWidgets.QHBoxLayout()
        toggle = QtWidgets.QToolButton()
        toggle.setText('视角与坐标轴')
        toggle.setCheckable(True)
        toggle.toggled.connect(self.view_options.setVisible)
        extras.addWidget(toggle)
        fit = QtWidgets.QToolButton()
        fit.setText('适配点云')
        fit.clicked.connect(self.canvas.fit_view)
        extras.addWidget(fit)
        if allow_large_view:
            large = QtWidgets.QToolButton()
            large.setText('展开 / 收起点云')
            large.clicked.connect(self.expand_button.toggle)
            extras.addWidget(large)
        self.focus_radius = QtWidgets.QDoubleSpinBox()
        self.focus_radius.setRange(1, 10000)
        self.focus_radius.setDecimals(1)
        self.focus_radius.setEnabled(False)
        self.focus_radius.valueChanged.connect(self.canvas.set_focus_radius)
        extras.addWidget(QtWidgets.QLabel('邻域半径'))
        extras.addWidget(self.focus_radius)
        extras.addWidget(QtWidgets.QLabel('点大小'))
        self.point_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size.setRange(5, 20)
        self.point_size.setValue(10)
        self.point_size.setMaximumWidth(95)
        self.point_size.valueChanged.connect(self.change_point_size)
        extras.addWidget(self.point_size)
        extras.addStretch(1)
        settings_layout.addLayout(extras)
        settings_layout.addWidget(self.view_options)
        self.view_options.hide()
        legend = QtWidgets.QHBoxLayout()
        self.category_checks = []
        for category, (label, color) in enumerate(zip(self.canvas.LABELS, self.canvas.COLORS)):
            check = QtWidgets.QCheckBox(label)
            check.setStyleSheet('color:%s;' % color)
            check.setChecked(True)
            check.toggled.connect(lambda checked, category=category: self.canvas.set_category_visible(category, checked))
            legend.addWidget(check)
            self.category_checks.append(check)
        legend.addStretch(1)
        settings_layout.addLayout(legend)
        self.reference_info = QtWidgets.QLabel('端点参考与实际样本分别显示；颜色只表示深度簇。')
        self.reference_info.setWordWrap(True)
        self.reference_info.setStyleSheet('color:#53677c;font-size:11px;')
        settings_layout.addWidget(self.reference_info)
        layout.addWidget(self.canvas, 1)
        self.sample_info = QtWidgets.QLabel('单击点云样本，可联动查看其原图像素、深度和 XYZ。')
        self.sample_info.setWordWrap(True)
        self.sample_info.setStyleSheet('font-size:11px;color:#53677c;')
        self.sample_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.canvas.sample_selected.connect(self.show_sample)
        layout.addWidget(self.sample_info)

    def focus_sample(self):
        if not self.canvas.focus_selected():
            self.navigation_info.setText('请先单击一个可见样本，再按 F 聚焦')
        self.canvas.setFocus()

    def update_navigation(self):
        center = '端点' if self.canvas.orbit_focus_id is None else '样本 %s' % self.canvas.orbit_focus_id
        self.navigation_info.setText('适配倍率 %.2f× · 绕%s · WASD 平移 · Shift 精调' % (self.canvas.zoom,center))

    def change_point_size(self, value):
        self.canvas.point_size_factor = value / 10.0
        self.canvas.update()

    def open_large_view(self):
        if not self.canvas.endpoint:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.setWindowTitle('端点 3D 点云 · 冻结数据独立视图')
        dialog.resize(1100, 850)
        layout = QtWidgets.QVBoxLayout(dialog)
        note = QtWidgets.QLabel('此视图保留打开时的端点数据；主工作台切换目标不会改变本图。')
        layout.addWidget(note)
        panel = DepthTerrainPanel(allow_large_view=False)
        panel.set_data(self.canvas.endpoint, self.canvas.depth, self.canvas.detection, self.canvas.intrinsics)
        panel.projection_combo.setCurrentIndex(self.projection_combo.currentIndex())
        panel.depth_mode_combo.setCurrentIndex(self.depth_mode_combo.currentIndex())
        panel.focus_combo.setCurrentIndex(self.focus_combo.currentIndex())
        panel.focus_radius.setValue(self.focus_radius.value())
        panel.point_size.setValue(self.point_size.value())
        layout.addWidget(panel)
        self.large_view = dialog
        dialog.show()

    def change_focus(self, index):
        mode = self.focus_combo.itemData(index)
        self.canvas.set_focus_mode(mode)
        self.focus_radius.blockSignals(True)
        self.focus_radius.setSuffix(' px' if mode == 'pixel' else ' mm')
        self.focus_radius.setValue(self.canvas.focus_radius_px if mode == 'pixel' else self.canvas.focus_radius_m * 1000)
        self.focus_radius.setEnabled(mode != 'all')
        self.focus_radius.blockSignals(False)

    def show_sample(self, sample):
        width = self.canvas.depth.shape[1]
        pixel_id = sample['pixel_id']
        self.sample_info.setText('%s · 像素 (%d, %d) · Z %.2f mm\n相机 XYZ (m): %s · 相对端点 (m): %s' % (
            sample['category'], pixel_id % width, pixel_id // width, sample['depth_m'] * 1000,
            ', '.join('%.5f' % x for x in sample['xyz_m']),
            ', '.join('%+.5f' % x for x in sample['relative_m'])))

    def set_data(self, endpoint, depth, detection=None, intrinsics=None):
        self.canvas.set_data(endpoint, depth, detection, intrinsics)
        self.change_focus(self.focus_combo.currentIndex())
        self.sample_info.setText('单击样本联动原图；显示抽样仅影响绘图，估深使用全部原始有效样本。')
        if endpoint and len(self.canvas.vertices):
            xyz = self.canvas.origin_xyz_m
            status = '已通过估深' if not self.canvas.origin_depth_fallback else '未验证候选（仅作显示参考）'
            self.reference_info.setText('%s · 参考相机 XYZ=(%.4f, %.4f, %.4f) m；红箭头指向原相机。' % (status, *xyz))
        else:
            self.reference_info.setText('当前端点没有有效点云；不生成虚构样本。')
        for category, check in enumerate(self.category_checks):
            check.setText('%s %d' % (self.canvas.LABELS[category], int(np.count_nonzero(self.canvas.vertex_categories == category))))
