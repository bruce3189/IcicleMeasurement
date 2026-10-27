import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_tool.core.project_store import ProjectStore, default_projects_root
from tests.support import temporary_directory


class ProjectStoreTests(unittest.TestCase):
    def test_project_round_trip_and_unique_directories(self):
        with temporary_directory() as root:
            video = root / "source.mp4"
            video.write_bytes(b"fixture")
            store = ProjectStore(root / "projects")
            first = store.create("同名项目", [str(video)])
            second = store.create("同名项目", [str(video)])
            self.assertNotEqual(first.root_path, second.root_path)
            loaded = store.load(Path(first.root_path) / "project.json")
            self.assertEqual(loaded.videos[0].path, str(video.resolve()))
            self.assertTrue(loaded.videos[0].available)
            self.assertEqual(json.loads((Path(first.root_path) / "project.json").read_text(encoding="utf-8"))["format_version"], 1)

    def test_missing_video_can_be_relinked(self):
        with temporary_directory() as root:
            video = root / "source.mp4"
            video.write_bytes(b"fixture")
            store = ProjectStore(root / "projects")
            project = store.create("project", [str(video)])
            video.unlink()
            loaded = store.load(project.root_path)
            self.assertFalse(loaded.videos[0].available)
            replacement = root / "replacement.mp4"
            replacement.write_bytes(b"fixture")
            store.relink_video(loaded, loaded.videos[0].id, str(replacement))
            self.assertTrue(loaded.videos[0].available)
            self.assertEqual(Path(loaded.videos[0].path), replacement.resolve())

    def test_frozen_project_root_is_next_to_executable(self):
        with temporary_directory() as root:
            executable = root / "DatasetTool.exe"
            with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(executable)):
                self.assertEqual(default_projects_root(), root / "projects")


if __name__ == "__main__":
    unittest.main()
