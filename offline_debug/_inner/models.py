"""Functions for serializing and reconstructing exceptions with their tracebacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _FrameData:
    """Serialized data for a single stack frame."""

    code: bytes
    globals: dict[str, Any]
    locals: dict[str, Any]
    lasti: int
    lineno: int
    stack_depth: int
    module_name: str | None = None


@dataclass
class _ExceptionData:
    """Serialized data for an exception and its traceback."""

    exc_pickle: bytes
    tb_frames: list[_FrameData]
    cause: _ExceptionData | None = None
    context: _ExceptionData | None = None
