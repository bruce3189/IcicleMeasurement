"""项目管理、视频预览和后台抽帧页面。"""

from pathlib import Path

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                             QLabel, QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QProgressBar, QInputDialog, QVBoxLayout,
                             QWidget)

from dataset_tool.core.project_store import ProjectStore
from dataset_tool.services.video_extractor import extract_video
from dataset_tool.ui.widgets.thumbnail_gallery import ThumbnailGallery
from dataset_tool.workers.operation_worker import OperationWorker


def _extract_project_videos(videos, run_root, target_fps, progress_callback=None, cancel_callback=None):
    result = []
    total = len(videos)
    for index, video in enumerate(videos):
        video_dir = run_root / "frames" / (video.id + "-" + Path(video.name).stem)
        video_dir.mkdir(parents=True, exist_ok=False)

        def report(event, offset=index):
            payload = dict(event)
            payload["video_index"] = offset + 1
            payload["video_total"] = total
            progress_callback(payload)

        metadata = extract_video(video.path, video_dir, target_fps, report, cancel_callback)
        result.append({"video_id": video.id, "image_count": metadata["image_count"], "metadata": metadata,
                       "output_dir": str(video_dir)})
    return result


class VideoPage(QWidget):
    def __init__(self, store=None, status_callback=None, parent=None):
        super().__init__(parent)
        self.store = store or ProjectStore()
        self.status_callback = status_callback or (lambda message: None)
        self.project = None
        self.run = None
        self.worker = None
        self.project_list = QListWidget()
        self.preview = QLabel("选择视频以预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(380, 240)
        self.preview.setStyleSheet("background:#20242b;color:#d0d5dd;border-radius:8px")
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.01, 240.0)
        self.fps.setDecimals(2)
        self.fps.setValue(2.0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.gallery = ThumbnailGallery()
        self.gallery.setMaximumHeight(180)
        self.gallery.image_selected.connect(self._show_gallery_image)

        new_button = QPushButton("新建项目并导入视频…")
        open_button = QPushButton("打开项目…")
        extract_button = QPushButton("开始抽帧")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        new_button.clicked.connect(self.create_project)
        open_button.clicked.connect(self.open_project)
        extract_button.clicked.connect(self.start_extract)
        self.cancel_button.clicked.connect(self.cancel_extract)
        self.project_list.currentItemChanged.connect(self._selection_changed)

        side = QVBoxLayout()
        side.addWidget(new_button)
        side.addWidget(open_button)
        side.addWidget(QLabel("项目视频"))
        side.addWidget(self.project_list, 1)
        settings = QFormLayout()
        settings.addRow("目标 FPS", self.fps)
        side.addLayout(settings)
        side.addWidget(extract_button)
        side.addWidget(self.cancel_button)
        side.addWidget(self.progress)

        right = QVBoxLayout()
        self.video_info = QLabel("原视频只记录路径，不会复制进项目。")
        self.video_info.setWordWrap(True)
        right.addWidget(self.preview, 1)
        right.addWidget(self.video_info)
        right.addWidget(QLabel("抽帧缩略图（双击打开所在文件夹）"))
        right.addWidget(self.gallery)

        layout = QHBoxLayout(self)
        layout.addLayout(side, 1)
        layout.addLayout(right, 3)

    def create_project(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择一个或多个视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if not paths:
            return
        name, accepted = QInputDialog.getText(self, "新建项目", "项目名称：")
        if not accepted:
            return
        try:
            self.project = self.store.create(name, paths)
            self._refresh_videos()
            self.status_callback("已创建项目：{}".format(self.project.name))
        except Exception as error:
            QMessageBox.critical(self, "创建项目失败", str(error))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", str(self.store.projects_root), "项目文件 (project.json)")
        if not path:
            return
        try:
            self.project = self.store.load(Path(path))
            self.fps.setValue(self.project.target_fps)
            self._refresh_videos()
            self.status_callback("已打开项目：{}".format(self.project.name))
        except Exception as error:
            QMessageBox.critical(self, "打开项目失败", str(error))

    def _refresh_videos(self):
        self.project_list.clear()
        if not self.project:
            return
        for video in self.project.videos:
            status = "" if video.available else "（路径失效）"
            item = QListWidgetItem(video.name + status)
            item.setData(Qt.UserRole, video.id)
            self.project_list.addItem(item)
        if self.project_list.count():
            self.project_list.setCurrentRow(0)
        images = []
        for run in self.project.runs:
            frames = Path(run.output_path) / "frames"
            if frames.is_dir():
                images.extend(str(p) for p in sorted(frames.rglob("*.jpg")))
        self.gallery.set_images(images)

    def _selection_changed(self, current, previous):
        if not current or not self.project:
            return
        video = next((item for item in self.project.videos if item.id == current.data(Qt.UserRole)), None)
        if not video:
            return
        if not Path(video.path).is_file():
            self.preview.setText("视频路径不可用，请重新关联视频文件。")
            self.video_info.setText(video.path)
            if QMessageBox.question(self, "视频路径失效", "是否现在重新选择这个视频文件？") == QMessageBox.Yes:
                path, _ = QFileDialog.getOpenFileName(self, "重新关联视频")
                if path:
                    try:
                        self.store.relink_video(self.project, video.id, path)
                        self._refresh_videos()
                    except Exception as error:
                        QMessageBox.warning(self, "重新关联失败", str(error))
            return
        cap = cv2.VideoCapture(video.path)
        try:
            if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            ok, frame = cap.read()
            video.fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
            video.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if ok:
                video.height, video.width = frame.shape[:2]
            else:
                video.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.store.save(self.project)
            self.video_info.setText("{} × {}，{} 帧，源 FPS：{}".format(video.width, video.height, video.frame_count, video.fps or "未知"))
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
                self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        finally:
            cap.release()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current = self.project_list.currentItem()
        if current:
            self._selection_changed(current, None)

    def start_extract(self):
        if not self.project or not self.project.videos:
            QMessageBox.information(self, "尚无项目", "请先新建项目并导入视频。")
            return
        if self.worker and self.worker.isRunning():
            return
        target_fps = self.fps.value()
        self.project.target_fps = target_fps
        self.run = self.store.new_run(self.project, "extract", "extraction_runs", target_fps)
        self.store.update_run(self.project, self.run, "running")
        run_root = Path(self.run.output_path)
        self.worker = OperationWorker(_extract_project_videos, self.project.videos, run_root, target_fps)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._extract_completed)
        self.worker.cancelled.connect(self._extract_cancelled)
        self.worker.failed.connect(self._extract_failed)
        self.worker.finished.connect(self._worker_finished)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def cancel_extract(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_callback("正在取消；已完成的图片会保留。")

    def _progress(self, event):
        total_videos = max(1, event.get("video_total", 1))
        video_index = event.get("video_index", 1)
        current, total = event.get("current", 0), max(1, event.get("total", 1))
        self.progress.setValue(int(((video_index - 1) + current / total) * 100 / total_videos))
        self.status_callback("{}：{}/{} 帧".format(event.get("filename", "抽帧"), current, total))

    def _extract_completed(self, result):
        count = sum(item["image_count"] for item in result)
        self.store.update_run(self.project, self.run, "completed", count)
        for item in result:
            video = next(v for v in self.project.videos if v.id == item["video_id"])
            for key in ("fps", "frame_count", "width", "height"):
                video_value = {"fps": item["metadata"]["source_fps"], "frame_count": item["metadata"]["total_frames"],
                               "width": item["metadata"]["width"], "height": item["metadata"]["height"]}[key]
                setattr(video, key, video_value)
        self.store.save(self.project)
        self._refresh_videos()
        self.progress.setValue(100)
        self.status_callback("抽帧完成，共生成 {} 张图片。".format(count))

    def _extract_cancelled(self, result):
        count = 0
        frames = Path(self.run.output_path) / "frames"
        if frames.exists():
            count = len(list(frames.rglob("*.jpg")))
        self.store.update_run(self.project, self.run, "cancelled", count)
        self._refresh_videos()
        self.status_callback("抽帧已取消，已生成的 {} 张图片已保留。".format(count))

    def _extract_failed(self, message):
        frames = Path(self.run.output_path) / "frames"
        count = len(list(frames.rglob("*.jpg"))) if frames.exists() else 0
        self.store.update_run(self.project, self.run, "failed", count, error=message)
        self.status_callback("抽帧失败：" + message)
        QMessageBox.critical(self, "抽帧失败", message)

    def _worker_finished(self):
        self.cancel_button.setEnabled(False)
        self.worker = None

    def _show_gallery_image(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.video_info.setText(path)
