"""服务共享的进度、取消和输出目录处理约定。"""

from pathlib import Path
from typing import Callable, Optional


class OperationCancelled(Exception):
    """由用户取消操作时抛出，保留已经生成的输出。"""


ProgressCallback = Optional[Callable[[dict], None]]
CancelCallback = Optional[Callable[[], bool]]


def prepare_empty_output(path) -> Path:
    output = Path(path)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("输出目录已存在且非空，为保护现有数据已停止：{}".format(output))
    else:
        output.mkdir(parents=True, exist_ok=True)
    return output


def check_cancel(cancel_callback: CancelCallback) -> None:
    if cancel_callback and cancel_callback():
        raise OperationCancelled("操作已由用户取消")


def emit_progress(callback: ProgressCallback, **payload) -> None:
    if callback:
        callback(payload)
