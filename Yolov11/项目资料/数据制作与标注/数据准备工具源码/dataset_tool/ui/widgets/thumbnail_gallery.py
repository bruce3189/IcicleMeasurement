"""简易缩略图列表。"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QListWidget, QListWidgetItem


class ThumbnailGallery(QListWidget):
    image_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QPixmap(112, 78).size())
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(8)
        self.itemClicked.connect(self._selected)
        self.itemDoubleClicked.connect(self._open_location)

    def set_images(self, paths):
        self.clear()
        for path in paths:
            item = QListWidgetItem(QIcon(str(path)), Path(path).name)
            item.setData(Qt.UserRole, str(path))
            self.addItem(item)

    def _selected(self, item):
        self.image_selected.emit(item.data(Qt.UserRole))

    def _open_location(self, item):
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(item.data(Qt.UserRole)).parent)))
