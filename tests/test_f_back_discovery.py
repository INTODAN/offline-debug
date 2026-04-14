"""Tests for the dynamic f_back offset discovery logic."""

import ctypes
from unittest.mock import patch

from offline_debug._inner.frame_c_api import _get_f_back_offset


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
    with patch("offline_debug._inner.frame_c_api._py_frame_new", return_value="not a frame"):
        offset = _get_f_back_offset()
        assert offset is None


def test_get_f_back_offset_exception_in_try() -> None:
    """Test when an exception occurs early in the discovery process."""
    with patch(
        "offline_debug._inner.frame_c_api._py_thread_state_get",
        side_effect=RuntimeError("thread error"),
    ):
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
    tstate = ctypes.pythonapi.PyThreadState_Get()
    code = compile("pass", "<dummy>", "exec")
    frame = ctypes.pythonapi.PyFrame_New(tstate, code, {}, {})

    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    # Force the loop to check an offset that is 0 but NOT the real f_back.

    # Let's try this:
    with (
        patch("offline_debug._inner.frame_c_api._py_frame_new", return_value=frame),
        patch("offline_debug._inner.frame_c_api.range", return_value=[ptr_size * 10]),
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
