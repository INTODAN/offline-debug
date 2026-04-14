# Define C API for frame linking
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
    """Get the Py_IncRef C function configured with ctypes."""
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
    ptr = ctypes.c_void_p.from_address(id(frame) + _f_back_offset)
    ptr.value = id(f_back)


@cache
def _get_f_back_offset() -> int | None:
    """Dynamically discover the memory offset of f_back in PyFrameObject."""
    try:
        # Compile a dummy code object that we can use to create a frame.
        code = compile("pass", "<discovery>", "exec")
        # Create a new, detached frame object using the C API.
        frame = create_frame(code=code, frame_globals={}, frame_locals={})

        # We need a target frame object to point to.
        target = sys._getframe()  # noqa: SLF001
        target_addr = id(target)

        # We scan the frame object's memory for the f_back pointer.
        # We cap the scan at the object's actual size to avoid out-of-bounds reads.
        limit = sys.getsizeof(frame)
        ptr_size = ctypes.sizeof(ctypes.c_void_p)

        # We start scanning after the PyObject header (refcnt + type).
        for offset in range(2 * ptr_size, limit - ptr_size + 1, ptr_size):
            try:
                # We use c_ssize_t to read the raw value at the offset.
                current_val = ctypes.c_ssize_t.from_address(id(frame) + offset).value
                # f_back is initially NULL (0) in a newly created frame.
                if current_val == 0:
                    ctypes.c_ssize_t.from_address(id(frame) + offset).value = target_addr
                    # If reading f_back via Python now returns our target, we found it.
                    if frame.f_back is target:
                        # Success, but we must restore 0 so we don't mess up refcounts
                        # when 'frame' is eventually garbage collected.
                        ctypes.c_ssize_t.from_address(id(frame) + offset).value = 0
                        return offset
                    # Restore to 0 if this wasn't the correct offset.
                    ctypes.c_ssize_t.from_address(id(frame) + offset).value = 0
            except (AttributeError, ValueError, TypeError, RuntimeError):
                continue
    except Exception:  # noqa: BLE001
        return None
    return None
