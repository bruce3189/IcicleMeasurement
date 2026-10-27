"""项目元数据模型。"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class SourceVideo:
    id: str
    path: str
    name: str
    available: bool = True
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @classmethod
    def from_path(cls, path: str) -> "SourceVideo":
        from pathlib import Path

        source = Path(path).expanduser().resolve()
        return cls(id=uuid4().hex, path=str(source), name=source.name, available=source.is_file())


@dataclass
class OperationRun:
    id: str
    kind: str
    status: str
    output_path: str
    created_at: str = field(default_factory=now_iso)
    finished_at: Optional[str] = None
    target_fps: Optional[float] = None
    image_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Project:
    id: str
    name: str
    root_path: str
    created_at: str = field(default_factory=now_iso)
    format_version: int = 1
    target_fps: float = 2.0
    videos: List[SourceVideo] = field(default_factory=list)
    runs: List[OperationRun] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Project":
        data = dict(payload)
        data["videos"] = [SourceVideo(**item) for item in data.get("videos", [])]
        data["runs"] = [OperationRun(**item) for item in data.get("runs", [])]
        return cls(**data)
