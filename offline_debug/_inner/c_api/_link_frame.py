"""
Link 2 frames together, so that a linked chain of frames will eventually create a traceback.

The native cpython api does not expose a way to link frames together
"""

from __future__ import annotations

import ctypes
import sys
from functools import cache
from typing import TYPE_CHECKING, Any

from ._create_frame import create_frame

if TYPE_CHECKING:
    import _ctypes
    from types import FrameType


@cache
def _get_py_incref() -> ctypes._NamedFuncPointer:
    """
    Get the Py_IncRef C function configured with ctypes.

    Py_IncRef increases the reference count of an object.
    We call this function when we want to "own" an object,
    meaning that the object won't be deleted while we are still using it.

    In our case, we use the function to hold the f_back reference to link between frames.
    """
    func: ctypes._NamedFuncPointer = ctypes.pythonapi.Py_IncRef
    func.argtypes = (ctypes.py_object,)
    func.restype = None

    def errcheck[T](
        result: T,
        _func: _ctypes.CFuncPtr,
        _args: tuple[Any, ...],
    ) -> T:  # pragma: no cover
        if result is not None:
            msg = f"Unexpected {result=}, expected None."
            raise TypeError(msg)
        return result

    func.errcheck = errcheck
    return func


def link_frame(frame: FrameType, f_back: FrameType) -> None:
    """Link a frame to its parent frame using the discovered offset."""
    _f_back_offset = _get_f_back_offset()
    if _f_back_offset is None:
        msg = "Failed discovering the offset for the f_back property."
        raise RuntimeError(msg)

    # In Python, setting f_back means the child frame now owns a reference
    # to the parent frame. We must increment the reference count of the
    # parent to reflect this.
    py_incref = _get_py_incref()
    py_incref(f_back)

    # Use ctypes to write the address of the back frame into the discovered offset.
    # We find the address of the f_back property, by finding the start of the frame struct,
    # Then moving to the f_back offset in the struct.
    ptr = ctypes.c_void_p.from_address(id(frame) + _f_back_offset)
    ptr.value = id(f_back)


@cache
def _get_f_back_offset() -> int | None:
    """
    Dynamically discover the memory offset of f_back in PyFrameObject.

    The offset for the f_back can change between python versions and operating systems,
    So we find the location of the f_back property dynamically.
    The general idea of the algorithm is to create a mock frame,
    And scanning its memory until we hit the f_back property field.
    We know we hit the f_back property by
    setting the value of each slot in the memeory to another frame, and checking if the
    f_back property was populated after changing that memory location.
    """
    try:
        # Compile an empty code object that we can use to create a frame.
        code = compile("pass", "<discovery>", "exec")

        # Create 2 frames that we can attempt linking between to find the f_back property.
        frame = create_frame(code=code, frame_globals={}, frame_locals={})
        f_back_frame = create_frame(code=code, frame_globals={}, frame_locals={})

        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        jump_size = ptr_size

        ref_count_size = ptr_size  # The amount of references to the object
        type_object_size = ptr_size
        frame_header_size = ref_count_size + type_object_size
        start = frame_header_size  # The header does not contain the f_back property

        frame_size = sys.getsizeof(frame)

        # We scan the memory ptr_size bytes forward, so we decrease it from the end.
        # Since we want the last memory slot to be scanned as well,
        # we add +1 because the for loop is exclusive to the end.
        end = frame_size - ptr_size + 1

        # We start scanning after the PyObject header (refcnt + type).
        for offset in range(start, end, jump_size):
            candidate_f_back_address = id(frame) + offset
            candidate_f_back_ptr = ctypes.c_ssize_t.from_address(candidate_f_back_address)

            # f_back is initially NULL (0) in a newly created frame,
            # since no other frames are linked to it.
            if candidate_f_back_ptr.value != 0:
                continue

            candidate_f_back_ptr.value = id(f_back_frame)
            # If reading f_back via Python now returns our target, we found it.
            if frame.f_back is f_back_frame:
                # Success, but we must restore 0 so we don't mess up refcounts
                # when 'frame' is eventually garbage collected.
                candidate_f_back_ptr.value = 0
                return offset
            # Restore to 0 if this wasn't the correct offset.
            candidate_f_back_ptr.value = 0

    except Exception:  # noqa: BLE001
        return None
    return None
