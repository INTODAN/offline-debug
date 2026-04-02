"""Functions for serializing and reconstructing exceptions with their tracebacks."""

from __future__ import annotations

import ctypes
import marshal
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

# Define C API for frame creation
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

_py_decref = ctypes.pythonapi.Py_DecRef
_py_decref.argtypes = (ctypes.py_object,)


def _get_f_back_offset() -> int | None:
    """Dynamically discover the memory offset of f_back in PyFrameObject."""
    try:
        tstate = _py_thread_state_get()
        # Compile a dummy code object that we can use to create a frame.
        code = compile("pass", "<discovery>", "exec")
        # Create a new, detached frame object using the C API.
        frame = _py_frame_new(tstate, code, {}, {})
        if not isinstance(frame, types.FrameType):
            return None

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


# Internal attributes that are either unpicklable or redundant in a new process.
# We exclude these specifically because they are automatically recreated
# when the new frame is initialized or when the module is imported.
_INTERNAL_ATTRIBUTES_TO_SKIP = ("__builtins__", "__doc__", "__loader__", "__package__", "__spec__")


@dataclass
class _FrameData:
    """Serialized data for a single stack frame."""

    code: bytes
    globals: dict[str, Any]
    locals: dict[str, Any]
    lasti: int
    lineno: int
    stack_depth: int


@dataclass
class _ExceptionData:
    """Serialized data for an exception and its traceback."""

    exc_pickle: bytes
    tb_frames: list[_FrameData]
    cause: _ExceptionData | None = None
    context: _ExceptionData | None = None


def _get_stack_depth(frame: types.FrameType) -> int:
    """Calculate the depth of the current stack frame."""
    depth = 0
    curr: types.FrameType | None = frame
    while curr:
        depth += 1
        curr = curr.f_back
    return depth


def _filter_dict(d: dict) -> dict:
    """Filter dictionary to include only picklable items."""
    result = {}
    for k, v in d.items():
        if k in _INTERNAL_ATTRIBUTES_TO_SKIP:
            continue
        try:
            # We must verify if the value is picklable because many globals
            # (like open file handles, database connections, or modules)
            # cannot be saved to disk.
            pickle.dumps(v)
            result[k] = v
        except Exception:  # noqa: BLE001
            result[k] = f"<unpicklable {type(v).__name__}: {v!r}>"
    return result


def _serialize_exc_data(exc: BaseException) -> _ExceptionData:
    """Recursively serialize exception data into dataclasses."""
    tb_frames: list[_FrameData] = []
    curr_tb = exc.__traceback__
    while curr_tb:
        f = curr_tb.tb_frame
        tb_frames.append(
            _FrameData(
                code=marshal.dumps(f.f_code),
                globals=_filter_dict(f.f_globals),
                locals=_filter_dict(f.f_locals),
                lasti=curr_tb.tb_lasti,
                lineno=curr_tb.tb_lineno,
                stack_depth=_get_stack_depth(f),
            )
        )
        curr_tb = curr_tb.tb_next

    try:
        exc_pickle = pickle.dumps(exc)
    except Exception:  # noqa: BLE001
        exc_pickle = pickle.dumps(
            RuntimeError(f"Unpicklable exception {type(exc).__name__}: {exc!s}")
        )

    return _ExceptionData(
        exc_pickle=exc_pickle,
        tb_frames=tb_frames,
        cause=_serialize_exc_data(exc.__cause__) if exc.__cause__ else None,
        context=_serialize_exc_data(exc.__context__) if exc.__context__ else None,
    )


def save_traceback(exc: BaseException, file_path: str | Path) -> None:
    """Serialize an exception and its traceback to a file."""
    data = _serialize_exc_data(exc)
    with Path(file_path).open("wb") as f:
        pickle.dump(data, f)


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
    exc = pickle.loads(data.exc_pickle)  # noqa: S301
    if not isinstance(exc, BaseException):
        msg = f"Expected BaseException, but got {type(exc).__name__}"
        raise TypeError(msg)

    tstate = _py_thread_state_get()

    reconstructed_frames: list[tuple[types.FrameType, _FrameData]] = []
    prev_frame: types.FrameType | None = None
    for f_data in data.tb_frames:
        code = marshal.loads(f_data.code)  # noqa: S302

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
        frame = _py_frame_new(tstate, code, f_data.globals, f_data.locals)
        if not isinstance(frame, types.FrameType):
            msg = f"Expected types.FrameType, but got {type(frame).__name__}"
            raise TypeError(msg)

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
