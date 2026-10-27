"""应用主窗口和模块导航。"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMainWindow, QStackedWidget, QVBoxLayout, QWidget)

from dataset_tool.modules.registry import build_modules
from dataset_tool.ui.pages.image_sample_page import ImageSamplePage
from dataset_tool.ui.pages.labeled_split_page import LabeledSplitPage
from dataset_tool.ui.pages.unavailable_page import UnavailablePage
from dataset_tool.ui.pages.video_page import VideoPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据集准备工具")
        self.resize(1280, 820)
        self.status = QLabel("就绪")
        self.video_page = VideoPage(status_callback=self.status.setText)
        self.labeled_page = LabeledSplitPage(status_callback=self.status.setText, project_page=self.video_page)
        self.sample_page = ImageSamplePage(status_callback=self.status.setText, project_page=self.video_page)
        self.annotation_page = UnavailablePage("标注（未接入）")
        self.training_page = UnavailablePage("训练（未接入）")
        self.modules = build_modules(self.video_page, self.labeled_page, self.sample_page, self.annotation_page)

        self.navigation = QListWidget()
        self.navigation.setMinimumWidth(190)
        self.stack = QStackedWidget()
        self.page_indices = []
        for module in self.modules:
            item = QListWidgetItem(module.title)
            item.setData(Qt.UserRole, module.key)
            self.navigation.addItem(item)
            page = self.training_page if module.key == "training" else module.page_factory
            self.page_indices.append(self.stack.addWidget(page))
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.navigation.setCurrentRow(0)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        title = QLabel("<b>数据集准备工具</b>")
        title.setStyleSheet("font-size:16px;padding:12px 4px")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(self.navigation, 1)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self.stack, 1)
        content_layout.addWidget(self.status)
        root = QHBoxLayout()
        root.addWidget(sidebar)
        root.addWidget(content, 1)
        host = QWidget()
        host.setLayout(root)
        self.setCentralWidget(host)
