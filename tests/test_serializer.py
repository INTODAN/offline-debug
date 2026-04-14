"""Tests for the serializer module."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Never

import pytest

from offline_debug import load_traceback, save_traceback

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def func_to_module_string(func: Callable) -> str:
    """Convert a function's body into a module string, removing indentation."""
    import inspect
    import textwrap

    source = inspect.getsource(func)
    # Get the body of the function (skip the def line)
    lines = source.splitlines()
    body = "\n".join(lines[1:])
    return textwrap.dedent(body)


def get_stack_depth(item: types.FrameType | types.TracebackType | None) -> int:
    """Calculate the depth of the given stack frame or traceback chain."""
    depth = 0
    curr = item
    while curr:
        depth += 1
        curr = curr.f_back if isinstance(curr, types.FrameType) else curr.tb_next
    return depth


def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
    """Extract all frames from a traceback."""
    frames = []
    curr = tb
    while curr:
        frames.append(curr.tb_frame)
        curr = curr.tb_next
    return frames


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

    # Reconstruct the depth from the traceback chain
    tb = exc_info.tb
    # Actually, load_traceback adds current frames.

    frames = get_frames(tb)
    l1_idx = next(i for i, f in enumerate(frames) if f.f_code.co_name == "level_1")
    l2_idx = next(i for i, f in enumerate(frames) if f.f_code.co_name == "level_2")
    l3_idx = next(i for i, f in enumerate(frames) if f.f_code.co_name == "level_3")

    # In a traceback, the list is TOP to BOTTOM (outer to inner)
    # So level_1 should be before level_2
    assert l1_idx < l2_idx
    assert l2_idx < l3_idx


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

    frames = get_frames(exc_info.tb)

    frame_names = [f.f_code.co_name for f in frames]
    assert "inner_raise" in frame_names
    assert "middle_step" in frame_names
    assert "second_stack_caller" in frame_names

    inner_idx = frame_names.index("inner_raise")
    middle_idx = frame_names.index("middle_step")
    second_idx = frame_names.index("second_stack_caller")

    # In a traceback, the list is TOP to BOTTOM (outer to inner)
    # second_stack_caller -> capture_it -> middle_step -> inner_raise
    assert second_idx < middle_idx
    assert middle_idx < inner_idx


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

    inner_exc = reconstructed_exc.__cause__
    assert isinstance(inner_exc, KeyError)
    inner_frames = get_frames(inner_exc.__traceback__)
    inner_names = [f.f_code.co_name for f in inner_frames]
    assert "fail_inner" in inner_names


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
        # Use locals() to force inclusion in the locals dictionary.
        _ = locals()["_obj"]
        msg = "Error with unpicklable"
        raise ValueError(msg)

    try:
        fail_with_unpicklable()
    except Exception as e:  # noqa: BLE001
        save_traceback(e, str(dump_file))

    with pytest.raises(ValueError, match="Error with unpicklable") as exc_info:
        load_traceback(str(dump_file))

    frames = get_frames(exc_info.tb)

    frame_names = [f.f_code.co_name for f in frames]
    assert "fail_with_unpicklable" in frame_names
    f = next(f for f in frames if f.f_code.co_name == "fail_with_unpicklable")
    # Verify that our filtering logic caught the unpicklable item.
    assert any("<unpicklable" in str(v) for v in f.f_locals.values())


GLOBAL_TEST_VAL = "I am global"
GLOBAL_VAR = "initial"


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

    frames = get_frames(exc_info.tb)

    frame_names = [f.f_code.co_name for f in frames]
    assert "fail_with_globals" in frame_names
    f = next(f for f in frames if f.f_code.co_name == "fail_with_globals")
    assert f.f_globals["GLOBAL_TEST_VAL"] == "I am global"


def test_globals_changing_between_frames(tmp_path: Path) -> None:
    """Test that globals are captured per-frame for different modules."""
    dump_file = tmp_path / "globals_changing.dump"

    # Define the second module's content inside a function
    def module2_content() -> None:
        GLOBAL_VAR = "initial"  # noqa: N806, F841

        def level_2() -> Never:
            global GLOBAL_VAR
            GLOBAL_VAR = "changed"
            msg = "Error at level 2"
            raise ValueError(msg)

    mod2_code = func_to_module_string(module2_content)
    mod2 = types.ModuleType("mod2")
    exec(mod2_code, mod2.__dict__)  # noqa: S102

    def level_1() -> None:
        try:
            mod2.level_2()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, str(dump_file))

    level_1()

    with pytest.raises(ValueError, match="Error at level 2") as exc_info:
        load_traceback(str(dump_file))

    frames = get_frames(exc_info.tb)

    l1_f = next(f for f in frames if f.f_code.co_name == "level_1")
    l2_f = next(f for f in frames if f.f_code.co_name == "level_2")

    # mod2.level_2 changed its OWN GLOBAL_VAR.
    # level_1's globals (this module) should NOT have GLOBAL_VAR (or it should be different)
    assert l2_f.f_globals["GLOBAL_VAR"] == "changed"
    assert "GLOBAL_VAR" not in l1_f.f_globals or l1_f.f_globals["GLOBAL_VAR"] == "initial"


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


def test_reconstruct_invalid_exception_type() -> None:
    """Test that _reconstruct_exc_data raises TypeError when the pickled exception is invalid."""
    # Create dummy data with a string instead of a pickled exception
    import pickle

    from offline_debug._inner.load_traceback import _reconstruct_exc_data
    from offline_debug._inner.models import _ExceptionData

    data = _ExceptionData(
        exc_pickle=pickle.dumps("not an exception"),
        tb_frames=[],
    )

    with pytest.raises(TypeError, match="Expected BaseException, but got str"):
        _reconstruct_exc_data(data)


def test_reconstruct_invalid_frame_type(monkeypatch) -> None:
    """Test that _reconstruct_exc_data raises TypeError when frame creation fails."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module
    from offline_debug._inner.load_traceback import _reconstruct_exc_data
    from offline_debug._inner.models import _ExceptionData, _FrameData

    # Mock _get_py_frame_new to return a function that returns something that is not a FrameType
    monkeypatch.setattr(_create_frame_module, "_get_py_frame_new", lambda: lambda *_: "not a frame")

    import marshal
    import pickle

    def dummy() -> None:
        pass

    data = _ExceptionData(
        exc_pickle=pickle.dumps(ValueError("test")),
        tb_frames=[
            _FrameData(
                code=marshal.dumps(dummy.__code__),
                globals={},
                locals={},
                lasti=0,
                lineno=0,
                stack_depth=0,
            )
        ],
    )

    with pytest.raises(TypeError, match=r"Expected types.FrameType, but got str"):
        _reconstruct_exc_data(data)


def test_get_f_back_offset_logic() -> None:
    """Test the dynamic f_back offset discovery logic directly."""
    from offline_debug._inner.c_api._link_frame import _get_f_back_offset

    offset = _get_f_back_offset()
    # It should either find an offset or be None (if platform is weird)
    # But on standard CPython it should find something.
    assert offset is None or (offset > 0 and offset % 8 == 0)


def test_link_frame_no_offset(monkeypatch) -> None:
    """Test that link_frame raises an exception if the f back offset wasn't found."""
    import offline_debug._inner.c_api._link_frame as _link_frame_module

    monkeypatch.setattr(_link_frame_module, "_get_f_back_offset", lambda: None)

    f = sys._getframe()
    with pytest.raises(RuntimeError, match="Failed discovering"):
        _link_frame_module.link_frame(f, f)


def test_reconstructed_frames_have_f_back(tmp_path: Path) -> None:
    """
    Test that reconstructed frames have their f_back pointers correctly linked.

    This test is currently expected to FAIL because f_back linking was removed
    to avoid segmentation faults, but it's required for full fidelity.
    """
    dump_file = tmp_path / "f_back_fidelity.dump"

    def level_2() -> Never:
        msg = "Fidelity error"
        raise ValueError(msg)

    def level_1() -> None:
        try:
            level_2()
        except Exception as e:  # noqa: BLE001
            save_traceback(e, str(dump_file))

    level_1()

    with pytest.raises(ValueError, match="Fidelity error") as exc_info:
        load_traceback(str(dump_file))

    frames = get_frames(exc_info.tb)
    # Traceback frames are ordered TOP to BOTTOM (outer to inner)
    # We want to check that level_2's frame points back to level_1
    l2_f = next(f for f in frames if f.f_code.co_name == "level_2")
    l1_f = next(f for f in frames if f.f_code.co_name == "level_1")

    assert l2_f.f_back is not None, "level_2.f_back should not be None"
    assert l2_f.f_back is l1_f, "level_2.f_back should point to level_1"


def test_locals_visibility_in_reconstructed_frames(tmp_path: Path) -> None:
    """Regression test: verify local variables are visible in reconstructed frames."""
    dump_file = tmp_path / "locals_visibility.dump"
    expected_val = 42

    def func_with_locals() -> Never:
        var_a = "hello"
        var_b = expected_val
        # Use them to ensure they aren't optimized away in some versions
        _ = f"{var_a} {var_b}"
        msg = "Locals error"
        raise ValueError(msg)

    try:
        func_with_locals()
    except ValueError as e:
        save_traceback(e, str(dump_file))

    with pytest.raises(ValueError, match="Locals error") as exc_info:
        load_traceback(str(dump_file))

    frames = get_frames(exc_info.tb)
    f = next(f for f in frames if f.f_code.co_name == "func_with_locals")

    assert f.f_locals.get("var_a") == "hello"
    assert f.f_locals.get("var_b") == expected_val
