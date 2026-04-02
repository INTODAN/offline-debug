"""Intended for manual use, run the test in debug mode and view the exceptions."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rich import traceback
from rich.console import Console

from offline_debug.serializer import load_traceback, save_traceback

global_variable = 1

console = Console(force_terminal=True)
traceback.install(show_locals=True, console=console)


def failure() -> None:
    """Raise a nested exception for testing."""
    _python_version = sys.version

    def exception_raising_func() -> None:
        local = "local"
        msg = f"exception {local}"
        raise ValueError(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_file = Path(tmpdir) / "exception.dump"
        try:
            exception_raising_func()
        except ValueError as e:
            save_traceback(e, dump_file)

        global global_variable
        global_variable += 1
        load_traceback(dump_file)


if __name__ == "__main__":
    failure()
