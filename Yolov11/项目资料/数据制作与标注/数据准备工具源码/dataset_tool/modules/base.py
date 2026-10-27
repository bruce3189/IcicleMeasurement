"""用于主导航的简单模块描述，不实现动态插件加载。"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ToolModule:
    key: str
    title: str
    page_factory: object
    available: bool = True
    note: Optional[str] = None
