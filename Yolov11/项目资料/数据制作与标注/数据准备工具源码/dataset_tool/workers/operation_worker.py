"""在 QThread 中执行服务，避免阻塞界面。"""

from threading import Event

from PyQt5.QtCore import QThread, pyqtSignal


class OperationWorker(QThread):
    progress = pyqtSignal(dict)
    completed = pyqtSignal(object)
    cancelled = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self._cancel_event = Event()

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        from dataset_tool.services.common import OperationCancelled

        self.kwargs["progress_callback"] = lambda data: self.progress.emit(data)
        self.kwargs["cancel_callback"] = self._cancel_event.is_set
        try:
            result = self.operation(*self.args, **self.kwargs)
            if self._cancel_event.is_set():
                self.cancelled.emit(result)
            else:
                self.completed.emit(result)
        except OperationCancelled:
            self.cancelled.emit(None)
        except Exception as error:
            self.failed.emit(str(error))
