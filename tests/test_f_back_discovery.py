"""Tests for the dynamic f_back offset discovery logic."""

import ctypes
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from offline_debug._inner.c_api._link_frame import _get_f_back_offset


@pytest.fixture(autouse=True)
def clear_f_back_offset_cache() -> Iterator[None]:
    _get_f_back_offset.cache_clear()
    yield
    _get_f_back_offset.cache_clear()


def test_get_f_back_offset_success() -> None:
    """Test successful discovery of f_back offset."""
    # We don't need to mock much here, just verify it returns a plausible offset.
    offset = _get_f_back_offset()
    assert isinstance(offset, int)
    assert offset > 0
    # On 64-bit CPython it's usually 16 or 24.
    assert offset % ctypes.sizeof(ctypes.c_void_p) == 0


def test_get_f_back_offset_not_a_frame() -> None:
    """Test when PyFrame_New returns something that is not a FrameType."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module

    with patch.object(
        _create_frame_module, "_get_py_frame_new", return_value=lambda *_: "not a frame"
    ):
        offset = _get_f_back_offset()
        assert offset is None


def test_get_f_back_offset_exception_in_try() -> None:
    """Test when an exception occurs early in the discovery process."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module

    mock_func = MagicMock(side_effect=RuntimeError("thread error"))
    with patch.object(_create_frame_module, "_get_py_thread_state_get", return_value=mock_func):
        offset = _get_f_back_offset()
        assert offset is None


def test_get_f_back_offset_discovery_failure() -> None:
    """Test when the discovery loop completes without finding the offset."""

    class MockValue:
        def __init__(self, val: int) -> None:
            self.value = val

    with patch("ctypes.c_ssize_t.from_address", return_value=MockValue(1)):
        offset = _get_f_back_offset()
        assert offset is None


def test_get_f_back_offset_wrong_offset_restoration() -> None:
    """Test that it restores 0 if the offset was wrong."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module
    import offline_debug._inner.c_api._link_frame as _link_frame_module

    tstate = ctypes.pythonapi.PyThreadState_Get()
    code = compile("pass", "<dummy>", "exec")
    frame = ctypes.pythonapi.PyFrame_New(tstate, code, {}, {})

    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    # Force the loop to check an offset that is 0 but NOT the real f_back.

    # Let's try this:
    with (
        patch.object(_create_frame_module, "_get_py_frame_new", return_value=lambda *_: frame),
        patch.object(_link_frame_module, "range", return_value=[ptr_size * 10]),
    ):  # Offset 80
        # Ensure offset 80 is 0
        ctypes.c_ssize_t.from_address(id(frame) + ptr_size * 10).value = 0
        offset = _get_f_back_offset()
        assert offset is None
        # Verify it was restored to 0
        assert ctypes.c_ssize_t.from_address(id(frame) + ptr_size * 10).value == 0


def test_get_f_back_offset_ctypes_error() -> None:
    """Test when ctypes.c_ssize_t.from_address raises an error."""
    with patch("ctypes.c_ssize_t.from_address", side_effect=ValueError("ctypes error")):
        offset = _get_f_back_offset()
        assert offset is None
