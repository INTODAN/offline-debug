"""
Intended for manual use, run the test in debug mode and view the exceptions and the function's locals
"""

import tempfile
from pathlib import Path

from offline_debug import save_traceback, load_traceback


def failure():
    def exception_raising_func() -> None:
        local = "local"
        raise ValueError(f"exception {local}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            exception_raising_func()
        except ValueError as e:
            save_traceback(e, Path(tmpdir) / "traceback.dump")
        load_traceback(Path(tmpdir) / "traceback.dump")


if __name__ == "__main__":
    failure()
