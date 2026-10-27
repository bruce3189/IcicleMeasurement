"""未接入功能的清晰占位页面。"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class UnavailablePage(QWidget):
    def __init__(self, title="功能未接入", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("{}\n当前版本仅预留模块接口。".format(title))
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px; color: #667085; padding: 48px;")
        layout.addWidget(label)
