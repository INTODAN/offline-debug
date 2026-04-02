"""Tests for the serializer module."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Never

import pytest

from offline_debug import load_traceback, save_traceback

if TYPE_CHECKING:
    import types
    from pathlib import Path


def get_stack_depth(frame: types.FrameType | None) -> int:
    """Calculate the depth of the given stack frame."""
    depth = 0
    curr = frame
    while curr:
        depth += 1
        curr = curr.f_back
    return depth


def test_stack_depth_preservation(tmp_path: Path) -> None:
    """Test that the stack depth is preserved during serialization."""
    dump_file = tmp_path / "depth.dump"

    original_depths = []

    def level_3() -> Never:
        f = sys._getframe()
        original_depths.append(get_stack_depth(f))
        msg = "Depth error"
        raise ValueError(msg)

    def level_2() -> None:
        f = sys._getframe()
        original_depths.append(get_stack_depth(f))
        level_3()

    def level_1() -> None:
        f = sys._getframe()
        original_depths.append(get_stack_depth(f))
        try:
            level_2()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, str(dump_file))

    level_1()

    with pytest.raises(ValueError, match="Depth error") as exc_info:
        load_traceback(str(dump_file))

    tb = exc_info.tb
    frames = []
    while tb:
        frames.append(tb.tb_frame)
        tb = tb.tb_next

    # We want to find level_1, level_2, level_3 in the reconstructed traceback
    l1_f = next(f for f in frames if f.f_code.co_name == "level_1")
    l2_f = next(f for f in frames if f.f_code.co_name == "level_2")
    l3_f = next(f for f in frames if f.f_code.co_name == "level_3")

    d1 = get_stack_depth(l1_f)
    d2 = get_stack_depth(l2_f)
    d3 = get_stack_depth(l3_f)

    # At minimum, verify they are linked correctly (depth increases by 1)
    assert d2 == d1 + 1
    assert d3 == d2 + 1

    # And verify they are NOT 1 (unless they were actually at depth 1)
    assert d1 > 1
    assert d2 > 1
    assert d3 > 1


def test_simple_exception_full_stack(tmp_path: Path) -> None:
    """Test serialization of a simple exception with a full stack."""
    dump_file = tmp_path / "simple.dump"

    def inner_raise() -> Never:
        _x = 10
        _y = "hello"
        msg = "Simple error"
        raise ValueError(msg)

    def middle_step() -> None:
        inner_raise()

    def capture_it() -> None:
        try:
            middle_step()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, str(dump_file))

    capture_it()
    assert dump_file.exists()

    # Now we call load_traceback from another stack
    def second_stack_caller() -> None:
        load_traceback(str(dump_file))

    with pytest.raises(ValueError, match="Simple error") as exc_info:
        second_stack_caller()

    tb = exc_info.tb
    frames = []
    while tb:
        frames.append(tb.tb_frame)
        tb = tb.tb_next

    frame_names = [f.f_code.co_name for f in frames]
    assert "inner_raise" in frame_names
    assert "middle_step" in frame_names
    assert "second_stack_caller" in frame_names

    inner_idx = frame_names.index("inner_raise")
    inner_frame = frames[inner_idx]
    assert inner_frame.f_back is not None
    assert inner_frame.f_back.f_code.co_name == "middle_step"


def test_chained_exceptions_stack(tmp_path: Path) -> None:
    """Test serialization of chained exceptions."""
    dump_file = tmp_path / "chained.dump"

    def fail_inner() -> Never:
        msg = "Inner key error"
        raise KeyError(msg)

    def fail_outer() -> None:
        try:
            fail_inner()
        except KeyError as e:
            msg = "Outer runtime error"
            raise RuntimeError(msg) from e

    try:
        fail_outer()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, str(dump_file))

    with pytest.raises(RuntimeError, match="Outer runtime error") as exc_info:
        load_traceback(str(dump_file))

    reconstructed_exc = exc_info.value
    assert isinstance(reconstructed_exc.__cause__, KeyError)

    def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
        f = []
        curr = tb
        while curr is not None:
            f.append(curr.tb_frame)
            curr = curr.tb_next
        return f

    outer_frames = get_frames(reconstructed_exc.__traceback__)
    outer_names = [f.f_code.co_name for f in outer_frames]
    assert "fail_outer" in outer_names

    fo_frame = next(f for f in outer_frames if f.f_code.co_name == "fail_outer")
    assert fo_frame.f_back is not None


def test_unpicklable_locals_verification(tmp_path: Path) -> None:
    """Test that unpicklable local variables are handled gracefully."""
    dump_file = tmp_path / "unpicklable.dump"

    class Unpicklable:
        def __reduce__(self) -> Never:
            msg = "Cannot pickle me"
            raise TypeError(msg)

        def __repr__(self) -> str:
            return "<Unpicklable Object>"

    def fail_with_unpicklable() -> Never:
        _obj = Unpicklable()
        msg = "Error with unpicklable"
        raise ValueError(msg)

    try:
        fail_with_unpicklable()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, str(dump_file))

    with pytest.raises(ValueError, match="Error with unpicklable") as exc_info:
        load_traceback(str(dump_file))

    tb = exc_info.tb
    frames = []
    while tb:
        frames.append(tb.tb_frame)
        tb = tb.tb_next

    frame_names = [f.f_code.co_name for f in frames]
    assert "fail_with_unpicklable" in frame_names
    f = next(f for f in frames if f.f_code.co_name == "fail_with_unpicklable")
    assert "_obj" in f.f_locals
    assert "<unpicklable Unpicklable: <Unpicklable Object>>" in f.f_locals["_obj"]


def test_global_variables_in_stack(tmp_path: Path) -> None:
    """Test that global variables are preserved in the stack."""
    dump_file = tmp_path / "globals.dump"

    def fail_with_globals() -> None:
        global GLOBAL_TEST_VAL  # noqa: PLW0602
        if GLOBAL_TEST_VAL == "I am global":
            msg = "Global test"
            raise ValueError(msg)

    try:
        fail_with_globals()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, str(dump_file))

    with pytest.raises(ValueError, match="Global test") as exc_info:
        load_traceback(str(dump_file))

    tb = exc_info.tb
    frames = []
    while tb:
        frames.append(tb.tb_frame)
        tb = tb.tb_next

    frame_names = [f.f_code.co_name for f in frames]
    assert "fail_with_globals" in frame_names
    f = next(f for f in frames if f.f_code.co_name == "fail_with_globals")
    assert f.f_globals["GLOBAL_TEST_VAL"] == "I am global"


GLOBAL_TEST_VAL = "I am global"


def test_unpicklable_exception_coverage(tmp_path: Path) -> None:
    """Test that unpicklable exceptions are handled by falling back to RuntimeError."""
    dump_file = tmp_path / "unpicklable_exc.dump"

    class UnpicklableError(Exception):
        def __reduce__(self) -> Never:
            msg = "Cannot pickle me"
            raise TypeError(msg)

    def raise_unpicklable() -> Never:
        msg = "Unpicklable"
        raise UnpicklableError(msg)

    try:
        raise_unpicklable()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, str(dump_file))

    with pytest.raises(RuntimeError, match="Unpicklable exception UnpicklableError: Unpicklable"):
        load_traceback(str(dump_file))


def test_typing_never() -> None:
    """Test that load_traceback is correctly annotated with Never."""
    import typing

    from offline_debug import load_traceback

    load_traceback_annotations = typing.get_type_hints(load_traceback)
    assert load_traceback_annotations["return"] is typing.Never


def test_load_invalid_object(tmp_path: Path) -> None:
    """Test that load_traceback raises TypeError when loading an invalid object."""
    import pickle

    dump_file = tmp_path / "invalid.dump"
    with dump_file.open("wb") as f:
        pickle.dump("not an ExceptionData object", f)

    with pytest.raises(TypeError, match="Expected _ExceptionData, but got str"):
        load_traceback(str(dump_file))


def test_load_non_existent_file() -> None:
    """Test that load_traceback raises FileNotFoundError when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_traceback("non_existent_file.dump")
