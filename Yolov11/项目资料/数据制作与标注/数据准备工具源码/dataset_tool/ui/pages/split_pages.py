"""两个独立数据集划分页面的共用表单基类。"""

from pathlib import Path
from uuid import uuid4

from PyQt5.QtWidgets import (QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QPushButton,
                             QProgressBar, QSpinBox, QVBoxLayout, QWidget)

from dataset_tool.core.project_store import ProjectStore
from dataset_tool.services.image_sampler import sample_images
from dataset_tool.services.labeled_splitter import split_labeled_dataset
from dataset_tool.workers.operation_worker import OperationWorker


class DirectoryField(QWidget):
    def __init__(self, title, directory=True, parent=None):
        super().__init__(parent)
        self.path = QLineEdit()
        button = QPushButton("浏览…")
        button.clicked.connect(lambda: self._browse(title, directory))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.path, 1)
        layout.addWidget(button)

    def _browse(self, title, directory):
        path = QFileDialog.getExistingDirectory(self, title) if directory else QFileDialog.getOpenFileName(self, title)[0]
        if path:
            self.path.setText(path)


class BaseSplitPage(QWidget):
    def __init__(self, kind, title, status_callback=None, project_page=None, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.project_page = project_page
        self.status_callback = status_callback or (lambda message: None)
        self.store = project_page.store if project_page else ProjectStore()
        self.run = None
        self.worker = None
        self.output = DirectoryField("选择输出目录")
        self.progress = QProgressBar()
        self.result = QLabel("输入目录中的文件会被复制，源文件保持不变。")
        self.result.setWordWrap(True)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.start_button = QPushButton("开始划分")
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.form = QFormLayout()
        self.form.addRow("输出目录（留空则使用当前项目的新运行目录）", self.output)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>{}</h2>".format(title)))
        layout.addLayout(self.form)
        layout.addWidget(self.result)
        layout.addWidget(self.progress)
        actions = QHBoxLayout()
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        layout.addLayout(actions)

    def _output_path(self):
        entered = self.output.path.text().strip()
        if entered:
            return Path(entered)
        project = self.project_page.project if self.project_page else None
        if project:
            self.run = self.store.new_run(project, self.kind, "datasets")
            return Path(self.run.output_path)
        path = self.store.projects_root / "standalone_runs" / (self.kind + "-" + uuid4().hex[:12])
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _start_worker(self, operation, *args, output_position=None):
        try:
            output = self._output_path()
            if self.run:
                self.store.update_run(self.project_page.project, self.run, "running")
            operation_args = list(args)
            if output_position is None:
                operation_args.append(output)
            else:
                operation_args.insert(output_position, output)
            self.worker = OperationWorker(operation, *operation_args)
            self.worker.progress.connect(self._progress)
            self.worker.completed.connect(self._completed)
            self.worker.cancelled.connect(self._cancelled)
            self.worker.failed.connect(self._failed)
            self.worker.finished.connect(self._finished)
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress.setValue(0)
            self.worker.start()
        except Exception as error:
            QMessageBox.critical(self, "无法开始", str(error))

    def start(self):
        raise NotImplementedError

    def _progress(self, event):
        self.progress.setValue(int(100 * event.get("current", 0) / max(1, event.get("total", 1))))
        self.status_callback("复制进度：{}/{}".format(event.get("current", 0), event.get("total", 0)))

    def _completed(self, result):
        count = result.get("total_pairs", result.get("selected_count", 0))
        if self.run:
            self.store.update_run(self.project_page.project, self.run, "completed", count, details={"counts": result.get("counts", {})})
        self.result.setText("处理完成。结果：{}\n各集合数量：{}{}".format(result["output_dir"], result["counts"],
                            "\n" + result["warning"] if result.get("warning") else ""))
        if result.get("missing_labels"):
            self.result.setText(self.result.text() + "\n缺少标签并跳过：{} 张。".format(len(result["missing_labels"])))
        if result.get("labels_copied") is False:
            self.result.setText(self.result.text() + "\n此流程只复制图片，没有复制标签。")
        self.status_callback("划分完成：" + result["output_dir"])

    def _cancelled(self, result):
        if self.run:
            images = Path(self.run.output_path) / "images"
            count = len(list(images.rglob("*"))) if images.exists() else 0
            self.store.update_run(self.project_page.project, self.run, "cancelled", count)
        self.result.setText("操作已取消；已生成的文件已保留。")

    def _failed(self, message):
        if self.run:
            images = Path(self.run.output_path) / "images"
            count = len(list(images.rglob("*"))) if images.exists() else 0
            self.store.update_run(self.project_page.project, self.run, "failed", count, error=message)
        self.result.setText("处理失败：" + message)
        QMessageBox.critical(self, "处理失败", message)

    def _finished(self):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    def cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)


class LabeledSplitPage(BaseSplitPage):
    def __init__(self, status_callback=None, project_page=None, parent=None):
        super().__init__("labeled-split", "图片和标签划分", status_callback, project_page, parent)
        self.images = DirectoryField("选择图片目录")
        self.labels = DirectoryField("选择标签目录")
        self.form.insertRow(0, "图片目录", self.images)
        self.form.insertRow(1, "标签目录", self.labels)
        self.train, self.val, self.test = [self._ratio(value) for value in (0.7, 0.2, 0.1)]
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        self.form.addRow("train / 训练", self.train)
        self.form.addRow("val / 验证", self.val)
        self.form.addRow("test / 测试", self.test)
        self.form.addRow("随机种子", self.seed)

    @staticmethod
    def _ratio(value):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    def start(self):
        if not self.images.path.text() or not self.labels.path.text():
            QMessageBox.information(self, "需要输入目录", "请选择图片目录和标签目录。")
            return
        self._start_worker(split_labeled_dataset, self.images.path.text(), self.labels.path.text(),
                           self.train.value(), self.val.value(), self.test.value(), self.seed.value(), output_position=2)


class ImageSamplePage(BaseSplitPage):
    def __init__(self, status_callback=None, project_page=None, parent=None):
        super().__init__("image-sample", "图片抽样划分", status_callback, project_page, parent)
        self.images = DirectoryField("选择图片目录")
        self.form.insertRow(0, "图片目录", self.images)
        self.sample = self._ratio(0.2)
        self.train = self._ratio(0.7)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        self.form.addRow("抽样比例", self.sample)
        self.form.addRow("抽样后 train 比例", self.train)
        self.form.addRow("随机种子", self.seed)
        self.result.setText("默认从图片中抽取 20%，按 70/30 划分；此页面不会读取或复制标签。")

    @staticmethod
    def _ratio(value):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    def start(self):
        if not self.images.path.text():
            QMessageBox.information(self, "需要输入目录", "请选择图片目录。")
            return
        self._start_worker(sample_images, self.images.path.text(), self.sample.value(),
                           self.train.value(), self.seed.value(), output_position=1)
