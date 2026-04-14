"""Load traceback object from a dump file."""

import ctypes
import marshal
import pickle
import sys
import types
from pathlib import Path
from types import CodeType
from typing import Never

from offline_debug._inner.c_api import (
    _py_incref,
    create_new_frame,
)
from offline_debug._inner.models import (
    _ExceptionData,
    _FrameData,
)


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


def _link_frame(frame: types.FrameType, back: types.FrameType) -> None:
    """Link a frame to its parent frame using the discovered offset."""
    if _F_BACK_OFFSET is None:
        return

    # In Python, setting f_back means the child frame now owns a reference
    # to the parent frame. We must increment the reference count of the
    # parent to reflect this.
    _py_incref(back)

    # Use ctypes to write the address of the back frame into the discovered offset.
    ptr = ctypes.c_void_p.from_address(id(frame) + _F_BACK_OFFSET)
    ptr.value = id(back)


def _reconstruct_exc_data(data: _ExceptionData) -> BaseException:
    """
    Recursively reconstruct an exception from its serialized data.

    Note on Python Locals:
    Python uses two ways to store local variables:
    1. "Slow" locals: A dictionary used for module-level code and class definitions.
    2. "Fast" locals: A fixed-size array used for functions. This is faster than
       dictionary lookups because variables are accessed by index.

    During reconstruction, we must explicitly synchronize these because PyFrame_New
    does not automatically populate the "fast" locals array from a dictionary.
    """
    exc: BaseException = pickle.loads(data.exc_pickle)  # noqa: S301
    if not isinstance(exc, BaseException):
        msg = f"Expected BaseException, but got {type(exc).__name__}"
        raise TypeError(msg)

    reconstructed_frames: list[tuple[types.FrameType, _FrameData]] = []
    prev_frame: types.FrameType | None = None
    for f_data in data.tb_frames:
        code: CodeType = marshal.loads(f_data.code)  # noqa: S302

        # In Python 3.11 and 3.12, accessing f_locals on a frame created via
        # PyFrame_New for optimized code (functions) causes a segmentation fault
        # because the internal 'fast' locals array is not initialized.
        # As a workaround, we create a 'non-optimized' version of the code object
        # by compiling a dummy string. This ensures the bytecode is safe
        # (no LOAD_FAST) while preserving metadata like name and filename.
        if sys.version_info < (3, 13):
            # A simple module-level code object never has fast locals.
            dummy_code = compile("", code.co_filename, "exec")
            code = dummy_code.replace(
                co_name=code.co_name,
                co_firstlineno=code.co_firstlineno,
                co_qualname=code.co_qualname,
            )

        # PyFrame_New returns a new reference to a PyFrameObject.
        frame: types.FrameType = create_new_frame(
            code=code, frame_globals=f_data.globals, frame_locals=f_data.locals
        )

        # In 3.13+, PEP 667 allows safe write-through access to locals.
        if sys.version_info >= (3, 13) and f_data.locals:
            frame.f_locals.update(f_data.locals)

        if prev_frame:
            _link_frame(frame, prev_frame)

        reconstructed_frames.append((frame, f_data))
        prev_frame = frame

    tb_next: types.TracebackType | None = None
    for frame, f_data in reversed(reconstructed_frames):
        tb = types.TracebackType(
            tb_next=tb_next,
            tb_frame=frame,
            tb_lasti=f_data.lasti,
            tb_lineno=f_data.lineno,
        )
        tb_next = tb

    exc = exc.with_traceback(tb_next)

    if data.cause:
        exc.__cause__ = _reconstruct_exc_data(data.cause)
    if data.context:
        exc.__context__ = _reconstruct_exc_data(data.context)

    return exc


def load_traceback(file_path: str | Path) -> Never:
    """Load an exception and its traceback from a file and raise it."""
    with Path(file_path).open("rb") as f:
        data = pickle.load(f)  # noqa: S301

    if not isinstance(data, _ExceptionData):
        msg = f"Expected _ExceptionData, but got {type(data).__name__}"
        raise TypeError(msg)

    exc = _reconstruct_exc_data(data)

    current_frames: list[types.FrameType] = []
    curr: types.FrameType | None = sys._getframe(1)  # noqa: SLF001
    while curr:
        current_frames.append(curr)
        curr = curr.f_back

    if exc.__traceback__ and current_frames:
        reconstructed_outer = exc.__traceback__.tb_frame
        _link_frame(reconstructed_outer, current_frames[0])

    tb_chain: types.TracebackType | None = exc.__traceback__
    for frame in current_frames:
        tb_chain = types.TracebackType(
            tb_next=tb_chain,
            tb_frame=frame,
            tb_lasti=frame.f_lasti,
            tb_lineno=frame.f_lineno,
        )

    exc = exc.with_traceback(tb_chain)
    raise exc
