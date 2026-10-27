import unittest
from pathlib import Path
import tempfile

from extract import main as extract_main
from make_xz import split_and_copy_images
from split import split_dataset


class CliWrapperTests(unittest.TestCase):
    def test_extract_cli_rejects_invalid_fps(self):
        with self.assertRaises(SystemExit) as error:
            extract_main(["--fps", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_wrappers_expose_shared_service_functions(self):
        self.assertTrue(callable(split_dataset))
        self.assertTrue(callable(split_and_copy_images))

    def test_cli_default_paths_are_script_relative(self):
        import extract
        import split
        import make_xz
        self.assertEqual(extract.Path(__file__).resolve().parents[1], Path(extract.__file__).resolve().parent)
        self.assertEqual(split.SCRIPT_DIR, Path(split.__file__).resolve().parent)
        self.assertEqual(make_xz.ROOT_DIR, Path(make_xz.__file__).resolve().parent)


if __name__ == "__main__":
    unittest.main()
