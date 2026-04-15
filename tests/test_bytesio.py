"""Tests for BytesIO support in save_traceback and load_traceback."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Never

import pytest

from offline_debug import load_traceback, save_traceback

if TYPE_CHECKING:
    import types


def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
    """Extract all frames from a traceback."""
    frames = []
    curr = tb
    while curr:
        frames.append(curr.tb_frame)
        curr = curr.tb_next
    return frames


def test_bytesio_roundtrip() -> None:
    """Test that saving to and loading from a BytesIO object works."""
    buffer = BytesIO()
    expected_val = 42

    def func_with_locals() -> Never:
        _var_a = "hello"
        _var_b = expected_val
        msg = "BytesIO error"
        raise ValueError(msg)

    try:
        func_with_locals()
    except ValueError as e:
        save_traceback(e, buffer)

    # Reset buffer for reading
    buffer.seek(0)

    with pytest.raises(ValueError, match="BytesIO error") as exc_info:
        load_traceback(buffer)

    frames = get_frames(exc_info.tb)
    f = next(f for f in frames if f.f_code.co_name == "func_with_locals")

    assert f.f_locals.get("_var_a") == "hello"
    assert f.f_locals.get("_var_b") == expected_val


def test_save_traceback_invalid_type() -> None:
    """Test that save_traceback raises TypeError for invalid file type."""
    with pytest.raises(TypeError, match="Unexpected type for file str"):
        save_traceback(ValueError("test"), "not a path or bytesio")  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]


def test_load_bytesio_invalid_object() -> None:
    """Test that load_traceback raises TypeError for invalid objects in BytesIO."""
    import pickle

    buffer = BytesIO()
    pickle.dump("not an ExceptionData object", buffer)
    buffer.seek(0)

    with pytest.raises(TypeError, match="Expected _ExceptionData, but got str"):
        load_traceback(buffer)
