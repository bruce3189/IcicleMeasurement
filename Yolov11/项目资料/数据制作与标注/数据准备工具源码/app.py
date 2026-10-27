"""数据集准备工具启动入口。"""

import sys


def main():
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError as error:
        raise SystemExit("当前 Python 环境缺少 PyQt5。请运行：python -m pip install -r requirements-gui.txt") from error
    from dataset_tool.ui.main_window import MainWindow

    application = QApplication(sys.argv)
    application.setApplicationName("数据集准备工具")
    window = MainWindow()
    window.show()
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
