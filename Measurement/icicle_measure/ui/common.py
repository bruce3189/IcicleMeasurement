"""Shared fonts, image view and algorithm configuration dialog."""
import json
import os
from pathlib import Path
from dataclasses import asdict
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..models import Config

STATUS = {'accepted': '通过', 'rejected': '拒绝', 'skipped': '跳过', 'cancelled': '已取消'}


def configure_font(app):
    # Offscreen Qt on Windows may not enumerate system fonts. Register the
    # installed CJK font explicitly so both GUI and saved previews have text.
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/msyh.ttc'
    family = 'Microsoft YaHei'
    if font_path.is_file():
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            family = families[0]
    app.setFont(QtGui.QFont(family, 9))


class ImageView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setBackgroundBrush(QtGui.QColor('#15212c'))
        self.setDragMode(self.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setMinimumSize(380, 300)

    def set_image(self, array, fit=False):
        array = np.ascontiguousarray(array)
        h, w = array.shape[:2]
        img = QtGui.QImage(array.data, w, h, array.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.item.setPixmap(QtGui.QPixmap.fromImage(img))
        self.scene().setSceneRect(0, 0, w, h)
        if fit:
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        scale = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(scale, scale)

    def mouseDoubleClickEvent(self, event):
        self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle('算法参数（深度单位 m）')
        self.resize(540, 700)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('全部参数可保存/加载；修改后须重新计算。初始阈值尚未经过真实相机标定。'))
        self.editor = QtWidgets.QPlainTextEdit(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        self.editor.setFont(QtWidgets.QApplication.font())
        layout.addWidget(self.editor)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.value = config

    def accept(self):
        try:
            self.value = Config.from_dict(json.loads(self.editor.toPlainText()))
        except (ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, '参数无效', str(exc))
            return
        super().accept()
