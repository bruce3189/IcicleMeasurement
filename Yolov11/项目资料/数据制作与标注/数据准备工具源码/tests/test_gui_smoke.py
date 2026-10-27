import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from dataset_tool.ui.main_window import MainWindow


class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_main_window_builds_all_navigation_pages(self):
        window = MainWindow()
        self.assertEqual(window.navigation.count(), 5)
        self.assertEqual(window.stack.count(), 5)
        self.assertEqual(window.navigation.currentRow(), 0)
        window.close()


if __name__ == "__main__":
    unittest.main()
