"""创建、保存和重新打开独立数据处理项目。"""

import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from .models import OperationRun, Project, SourceVideo, now_iso


def default_projects_root() -> Path:
    """源码运行时放在仓库旁；打包运行时放在 exe 旁，避开 onefile 临时目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "projects"
    return Path(__file__).resolve().parents[2] / "projects"


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip()).strip(" ._")
    return cleaned[:80] or "未命名项目"


class ProjectStore:
    def __init__(self, projects_root: Optional[Path] = None):
        self.projects_root = Path(projects_root) if projects_root else default_projects_root()

    def create(self, name: str, video_paths: Iterable[str]) -> Project:
        videos = [SourceVideo.from_path(path) for path in video_paths]
        if not videos:
            raise ValueError("请至少选择一个视频文件")
        missing = [video.path for video in videos if not video.available]
        if missing:
            raise FileNotFoundError("视频文件不存在：" + ", ".join(missing))
        project_id = uuid4().hex[:12]
        project_root = self.projects_root / (project_id + "-" + safe_name(name))
        project_root.mkdir(parents=True, exist_ok=False)
        project = Project(id=project_id, name=name.strip() or "未命名项目", root_path=str(project_root), videos=videos)
        self.save(project)
        return project

    def load(self, project_json: Path) -> Project:
        path = Path(project_json)
        if path.is_dir():
            path = path / "project.json"
        with path.open("r", encoding="utf-8") as stream:
            project = Project.from_dict(json.load(stream))
        for video in project.videos:
            video.available = Path(video.path).is_file()
        return project

    def save(self, project: Project) -> None:
        root = Path(project.root_path)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "project.json"
        temporary = root / "project.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(project.to_dict(), stream, ensure_ascii=False, indent=2)
        temporary.replace(target)

    def relink_video(self, project: Project, video_id: str, new_path: str) -> None:
        source = Path(new_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("所选视频文件不存在")
        video = next((item for item in project.videos if item.id == video_id), None)
        if video is None:
            raise KeyError("项目中找不到该视频")
        video.path = str(source)
        video.name = source.name
        video.available = True
        self.save(project)

    def new_run(self, project: Project, kind: str, relative_parent: str, target_fps=None) -> OperationRun:
        run_id = uuid4().hex[:12]
        run_root = Path(project.root_path) / relative_parent / (kind + "-" + run_id)
        run_root.mkdir(parents=True, exist_ok=False)
        run = OperationRun(id=run_id, kind=kind, status="queued", output_path=str(run_root), target_fps=target_fps)
        project.runs.append(run)
        self.save(project)
        return run

    def update_run(self, project: Project, run: OperationRun, status: str, image_count=None, error=None, details=None) -> None:
        run.status = status
        if image_count is not None:
            run.image_count = image_count
        run.error = error
        if details is not None:
            run.details.update(details)
        if status in ("completed", "cancelled", "failed"):
            run.finished_at = now_iso()
        self.save(project)
