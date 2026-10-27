import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_tool.services.video_extractor import extract_video
from tests.support import temporary_directory


class FakeCapture:
    def __init__(self, path):
        self.frames = [object() for _ in range(6)]
        self.orientation_auto = 0

    def isOpened(self):
        return True

    def get(self, key):
        import cv2
        if key == cv2.CAP_PROP_ORIENTATION_META:
            return 90.0
        if key == cv2.CAP_PROP_ORIENTATION_AUTO:
            return self.orientation_auto
        values = {cv2.CAP_PROP_FPS: 5.0, cv2.CAP_PROP_FRAME_COUNT: 6,
                cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48}
        return values[key]

    def set(self, key, value):
        import cv2
        if key == cv2.CAP_PROP_ORIENTATION_AUTO:
            self.orientation_auto = value
            return True
        return False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


class VideoExtractorTests(unittest.TestCase):
    def test_target_fps_sampling_and_nonempty_guard(self):
        with temporary_directory() as root:
            source = root / "sample.avi"
            source.write_bytes(b"fixture")
            output = root / "frames"

            def fake_write(path, frame):
                Path(path).write_bytes(b"frame")
                return True

            with patch("dataset_tool.services.video_extractor.cv2.VideoCapture", FakeCapture), \
                 patch("dataset_tool.services.video_extractor.cv2.imwrite", fake_write):
                result = extract_video(source, output, 2)
                self.assertEqual(result["image_count"], 3)
                self.assertEqual(result["orientation_degrees"], 90.0)
                self.assertTrue(result["orientation_auto"])
                self.assertEqual(len(list(output.glob("*.jpg"))), 3)
                with self.assertRaises(FileExistsError):
                    extract_video(source, output, 1)

    def test_invalid_fps_is_rejected(self):
        with temporary_directory() as root:
            source = root / "fixture.avi"
            source.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                extract_video(source, root / "out", 0)


if __name__ == "__main__":
    unittest.main()
