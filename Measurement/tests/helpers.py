"""Shared temporary directories with inherited Windows workspace permissions."""
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


TEST_TEMP = Path(__file__).resolve().parents[1] / 'outputs' / 'test_tmp'
TEST_TEMP.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory():
    # Windows sandbox cannot re-open Python's private-ACL mkdtemp directories.
    # Normal inherited workspace permissions are sufficient for test fixtures.
    path = TEST_TEMP / ('case_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield str(path)
    finally:
        assert path.resolve().parent == TEST_TEMP.resolve()
        shutil.rmtree(str(path))
