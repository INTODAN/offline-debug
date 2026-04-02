"""Functions for serializing and reconstructing exceptions with their tracebacks."""

from __future__ import annotations

import ctypes
import marshal
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Never

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


class _PyFrameObject(ctypes.Structure):
    """
    CPython's internal representation of a stack frame (PyFrameObject).

    We use this structure to manually link reconstructed frames by modifying
    the f_back pointer, which Python's high-level API does not allow.
    """

    _fields_: ClassVar[list[tuple[str, Any]]] = [
        # Memory offset to skip the PyObject header (ob_refcnt and ob_type).
        # This is necessary so that ctypes knows the exact position of f_back in memory.
        (
            "offset_buffer",
            ctypes.c_byte * (ctypes.sizeof(ctypes.c_ssize_t) + ctypes.sizeof(ctypes.c_void_p)),
        ),
        ("f_back", ctypes.c_void_p),  # Pointer to the previous frame in the call stack.
    ]


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

        # PyFrame_New returns a new reference to a PyFrameObject.
        # We pass empty locals and update them afterward because PyFrame_New
        # does not correctly initialize "fast" locals from a dictionary.
        frame = _py_frame_new(tstate, code, f_data.globals, {})
        if not isinstance(frame, types.FrameType):
            msg = f"Expected types.FrameType, but got {type(frame).__name__}"
            raise TypeError(msg)

        if f_data.locals:
            frame.f_locals.update(f_data.locals)

        # Stitch the frames together to reconstruct the full stack trace.
        # This allows tools like pdb or IDE debuggers to navigate up and down
        # the reconstructed stack.
        if prev_frame:
            frame_ptr = _PyFrameObject.from_address(id(frame))
            frame_ptr.f_back = id(prev_frame)

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
        caller_frame = current_frames[0]

        frame_ptr = _PyFrameObject.from_address(id(reconstructed_outer))
        frame_ptr.f_back = id(caller_frame)

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
