"""Tests for module name preservation in save_traceback and load_traceback."""

from __future__ import annotations

import types
from io import BytesIO

import pytest

from offline_debug import load_traceback, save_traceback


def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
    """Extract all frames from a traceback."""
    frames = []
    curr = tb
    while curr:
        frames.append(curr.tb_frame)
        curr = curr.tb_next
    return frames


def test_module_name_preservation_from_spec() -> None:
    """Test that the real module name is preserved via __spec__.name if __name__ is __main__."""
    buffer = BytesIO()
    real_module_name = "my_package.my_module"

    # We simulate a frame where __name__ is "__main__" but __spec__.name is the real module name
    # (standard behavior for `python -m my_package.my_module`)
    class MockSpec:
        def __init__(self, name: str) -> None:
            self.name = name

    # Define a function to be executed in a specific globals context
    def fail_func() -> None:
        msg = "Module name error"
        raise ValueError(msg)

    custom_globals = {
        "__name__": "__main__",
        "__spec__": MockSpec(real_module_name),
        "__builtins__": __builtins__,
    }

    # Execute the function with our custom globals
    # We use a wrapper to capture the exception
    try:
        # Create a new function object with the same code but different globals
        # This is how we simulate it being part of another module
        new_func = types.FunctionType(fail_func.__code__, custom_globals, "fail_func")
        new_func()
    except ValueError as e:
        save_traceback(e, buffer)

    buffer.seek(0)

    with pytest.raises(ValueError, match="Module name error") as exc_info:
        load_traceback(buffer)

    frames = get_frames(exc_info.tb)
    f = next(f for f in frames if f.f_code.co_name == "fail_func")

    # Verify that __name__ in the reconstructed frame's globals is the real module name
    assert f.f_globals.get("__name__") == real_module_name


def test_module_name_preservation_normal() -> None:
    """Test that __name__ is preserved normally when it's not __main__."""
    buffer = BytesIO()
    module_name = "some_other_module"

    def fail_func() -> None:
        msg = "Normal name error"
        raise ValueError(msg)

    custom_globals = {
        "__name__": module_name,
        "__builtins__": __builtins__,
    }

    try:
        new_func = types.FunctionType(fail_func.__code__, custom_globals, "fail_func")
        new_func()
    except ValueError as e:
        save_traceback(e, buffer)

    buffer.seek(0)

    with pytest.raises(ValueError, match="Normal name error") as exc_info:
        load_traceback(buffer)

    frames = get_frames(exc_info.tb)
    f = next(f for f in frames if f.f_code.co_name == "fail_func")

    assert f.f_globals.get("__name__") == module_name
