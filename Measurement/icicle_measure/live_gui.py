"""Small live acquisition window. The main workspace always shows a frozen frame."""
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from .capture import CaptureRunner, RealSenseSource
from .pose import PoseConfig, default_weights


class CameraDialog(QtWidgets.QDialog):
    measure_requested = QtCore.pyqtSignal(object, object)

    def __init__(self, can_process, parent=None, source_factory=RealSenseSource):
        super().__init__(parent)
        self.setWindowTitle('D435i · RGB-D 采集与测量')
        self.resize(820, 680)
        self.can_process, self.source_factory = can_process, source_factory
        self.runner = self.packet = None
        self.last_submitted = None
        self.next_inference = 0.0
        self.closing = False
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('本窗为实时预览；测距工作台显示对应帧的冻结结果。连续模式只处理最新帧，不承诺每帧测量。')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.preview = QtWidgets.QLabel('连接 D435i 后显示预览')
        self.preview.setMinimumSize(480, 300)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet('background:#142332;color:#d8e5f4;border-radius:8px;')
        layout.addWidget(self.preview, 1)
        form = QtWidgets.QFormLayout()
        weights_row = QtWidgets.QHBoxLayout()
        self.weights = QtWidgets.QLineEdit(default_weights())
        browse = QtWidgets.QPushButton('选择权重')
        browse.clicked.connect(self.browse_weights)
        weights_row.addWidget(self.weights, 1)
        weights_row.addWidget(browse)
        form.addRow('两点 YOLO11 Pose：', weights_row)
        thresholds = QtWidgets.QHBoxLayout()
        self.box_conf = QtWidgets.QDoubleSpinBox()
        self.point_conf = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self.box_conf, .25), (self.point_conf, .3)):
            spin.setRange(0, 1)
            spin.setSingleStep(.05)
            spin.setValue(value)
        thresholds.addWidget(QtWidgets.QLabel('框置信度'))
        thresholds.addWidget(self.box_conf)
        thresholds.addWidget(QtWidgets.QLabel('端点置信度'))
        thresholds.addWidget(self.point_conf)
        self.order = QtWidgets.QComboBox()
        self.order.addItems(['模型点序：尖端 → 根部', '模型点序：根部 → 尖端'])
        thresholds.addWidget(self.order)
        form.addRow('检测设置：', thresholds)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        self.connect_button = QtWidgets.QPushButton('连接相机')
        self.stop_button = QtWidgets.QPushButton('停止采集')
        self.snapshot_button = QtWidgets.QPushButton('抓拍并测量')
        self.continuous = QtWidgets.QCheckBox('持续测量（CPU）')
        self.connect_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.snapshot_button.clicked.connect(self.submit)
        for widget in (self.connect_button, self.stop_button, self.snapshot_button, self.continuous):
            buttons.addWidget(widget)
        layout.addLayout(buttons)
        self.status = QtWidgets.QLabel('尚未连接；深度默认不补洞、不跨物体边缘平滑。')
        self.status.setWordWrap(True)
        self.result_label = QtWidgets.QLabel('尚无测距结果')
        self.result_label.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(self.result_label)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)
        self.poll()

    def browse_weights(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择两关键点 Pose 权重', '', 'PyTorch (*.pt)')
        if path:
            self.weights.setText(path)

    def start_capture(self):
        if self.runner and self.runner.is_alive():
            return
        self.packet, self.last_submitted = None, None
        self.preview.setText('正在连接 D435i…')
        self.status.setText('正在启动 640×480 / 30 FPS RGB-D 采集…')
        self.runner = CaptureRunner(self.source_factory)
        self.runner.start()
        self.poll()

    def stop_capture(self):
        self.continuous.setChecked(False)
        if self.runner:
            self.runner.stop()
        self.snapshot_button.setEnabled(False)

    def poll(self):
        running = bool(self.runner and self.runner.is_alive())
        stopping = bool(self.runner and self.runner.stop_event.is_set())
        self.connect_button.setEnabled(not running)
        self.stop_button.setEnabled(running and not stopping)
        self.snapshot_button.setEnabled(running and not stopping and self.packet is not None and self.can_process())
        self.continuous.setEnabled(running and not stopping)
        if self.runner:
            packet = self.runner.buffer.take()
            if packet is not None and running and not stopping:
                self.packet = packet
                rgb = packet.frame.rgb
                image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
                self.preview.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                    self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                self.status.setText('采集帧 #%d · 设备时间 %.1f ms · %s · 预览合并 %d 帧' % (
                    packet.sequence, packet.timestamp_ms, packet.captured_at, self.runner.buffer.overwritten))
            if not running and self.runner.error:
                self.continuous.setChecked(False)
                self.status.setText('采集失败：' + self.runner.error)
            elif not running and stopping:
                self.status.setText('采集已停止，工作台保留已完成的冻结结果。')
        if running and not stopping and self.continuous.isChecked() and self.can_process() and time.monotonic() >= self.next_inference:
            if self.packet is not None and self.packet.sequence != self.last_submitted:
                self.submit()
        if self.closing and not running:
            self.closing = False
            self.close()
            if self.parent() is not None and self.parent().close_pending:
                self.parent().close()

    def submit(self):
        if self.packet is None or not self.can_process():
            return
        try:
            config = PoseConfig(self.weights.text(), self.box_conf.value(), self.point_conf.value(),
                                reversed_order=self.order.currentIndex() == 1)
            config.validate()
        except ValueError as exc:
            self.processing_failed(str(exc))
            return
        self.last_submitted = self.packet.sequence
        self.next_inference = time.monotonic() + .25
        self.result_label.setText('正在处理帧 #%d；主工作台将显示该帧的 RGB、深度和检测结果。' % self.packet.sequence)
        self.measure_requested.emit(self.packet, config)

    def processing_finished(self, sequence, completed_at):
        self.result_label.setText('工作台结果：帧 #%d · 完成时间 %s（采集帧时间见结果来源记录）' % (sequence, completed_at))

    def processing_failed(self, reason):
        self.continuous.setChecked(False)
        self.result_label.setText('检测/测距失败，持续测量已暂停：' + reason)

    def closeEvent(self, event):
        if self.runner and self.runner.is_alive():
            self.closing = True
            self.stop_capture()
            event.ignore()
        else:
            event.accept()
