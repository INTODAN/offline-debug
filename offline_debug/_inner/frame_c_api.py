# Define C API for frame creation

import ctypes
import sys
from types import CodeType, FrameType
from typing import Any

_py_frame_new = ctypes.pythonapi.PyFrame_New
_py_frame_new.argtypes = (
    ctypes.c_void_p,  # PyThreadState *tstate
    ctypes.py_object,  # PyCodeObject *code
    ctypes.py_object,  # PyObject *globals
    ctypes.py_object,  # PyObject *locals
)
_py_frame_new.restype = ctypes.py_object

_py_thread_state_get = ctypes.pythonapi.PyThreadState_Get
_py_thread_state_get.restype = ctypes.c_void_p

_py_incref = ctypes.pythonapi.Py_IncRef
_py_incref.argtypes = (ctypes.py_object,)


def create_new_frame(
    code: CodeType,
    frame_globals: dict[str, Any],
    frame_locals: dict[str, Any],
    thread_state: int | None = None,
) -> FrameType:
    if thread_state is None:
        thread_state: int = _py_thread_state_get()
    frame: FrameType = _py_frame_new(thread_state, code, frame_globals, frame_locals)
    if not isinstance(frame, FrameType):
        msg = f"Expected types.FrameType, but got {type(frame).__name__}"
        raise TypeError(msg)
    return frame


def link_frame(frame: FrameType, f_back: FrameType) -> None:
    """Link a frame to its parent frame using the discovered offset."""
    if _F_BACK_OFFSET is None:
        msg = "Failed discovering the offset for the f_back property."
        raise RuntimeError(msg)

    # In Python, setting f_back means the child frame now owns a reference
    # to the parent frame. We must increment the reference count of the
    # parent to reflect this.
    _py_incref(f_back)

    # Use ctypes to write the address of the back frame into the discovered offset.
    ptr = ctypes.c_void_p.from_address(id(frame) + _F_BACK_OFFSET)
    ptr.value = id(f_back)


def _get_f_back_offset() -> int | None:
    """Dynamically discover the memory offset of f_back in PyFrameObject."""
    try:
        # Compile a dummy code object that we can use to create a frame.
        code = compile("pass", "<discovery>", "exec")
        # Create a new, detached frame object using the C API.
        frame = create_new_frame(code=code, frame_globals={}, frame_locals={})

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


_F_BACK_OFFSET = _get_f_back_offset()
