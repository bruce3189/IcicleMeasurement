"""创建 ACL 继承当前工作区的隔离测试目录。"""

from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from uuid import uuid4


@contextmanager
def temporary_directory():
    root = Path.cwd() / (".test-fixture-" + uuid4().hex[:12])
    root.mkdir()
    try:
        yield root
    finally:
        rmtree(str(root))
