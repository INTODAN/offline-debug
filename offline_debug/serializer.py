from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Never, cast
import pickle
import marshal
import types
import ctypes
import sys

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

# Constants for serialization
_INTERNAL_ATTRIBUTES = ("__builtins__", "__doc__", "__loader__", "__package__", "__spec__")


@dataclass
class _FrameData:
    code: bytes
    globals: Dict[str, Any]
    locals: Dict[str, Any]
    lasti: int
    lineno: int
    stack_depth: int


@dataclass
class _ExceptionData:
    exc_pickle: bytes
    tb_frames: List[_FrameData]
    cause: Optional["_ExceptionData"] = None
    context: Optional["_ExceptionData"] = None


# Define ctypes structure to access f_back in PyFrameObject
class _PyObject(ctypes.Structure):
    _fields_ = [("ob_refcnt", ctypes.c_ssize_t), ("ob_type", ctypes.c_void_p)]


class _PyFrameObject(ctypes.Structure):
    _fields_ = [
        ("ob_base", _PyObject),
        ("f_back", ctypes.c_void_p),  # Pointer to previous frame
    ]


def _get_stack_depth(frame: types.FrameType) -> int:
    depth = 0
    curr: Optional[types.FrameType] = frame
    while curr:
        depth += 1
        curr = curr.f_back
    return depth


def _filter_dict(d: dict) -> dict:
    """Filter dictionary to include only picklable items."""
    result = {}
    for k, v in d.items():
        if k in _INTERNAL_ATTRIBUTES:
            continue
        try:
            pickle.dumps(v)
            result[k] = v
        except Exception:
            result[k] = f"<unpicklable {type(v).__name__}: {repr(v)}>"
    return result


def _serialize_exc_data(exc: BaseException) -> _ExceptionData:
    tb_frames: List[_FrameData] = []
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
    except Exception:
        exc_pickle = pickle.dumps(
            RuntimeError(f"Unpicklable exception {type(exc).__name__}: {str(exc)}")
        )

    return _ExceptionData(
        exc_pickle=exc_pickle,
        tb_frames=tb_frames,
        cause=_serialize_exc_data(exc.__cause__) if exc.__cause__ else None,
        context=_serialize_exc_data(exc.__context__) if exc.__context__ else None,
    )


def save_traceback(exc: Exception, file_path: str | Path):
    """Serialize an exception and its traceback to a file."""
    data = _serialize_exc_data(exc)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def _reconstruct_exc_data(data: _ExceptionData) -> Exception:
    exc = cast(Exception, pickle.loads(data.exc_pickle))

    tstate = _py_thread_state_get()

    reconstructed_frames: List[tuple[types.FrameType, _FrameData]] = []
    prev_frame: Optional[types.FrameType] = None
    for f_data in data.tb_frames:
        code = marshal.loads(f_data.code)

        # PyFrame_New returns a new reference to a PyFrameObject
        frame = cast(types.FrameType, _py_frame_new(tstate, code, f_data.globals, {}))

        if f_data.locals:
            frame.f_locals.update(f_data.locals)

        if prev_frame:
            frame_ptr = _PyFrameObject.from_address(id(frame))
            frame_ptr.f_back = id(prev_frame)

        reconstructed_frames.append((frame, f_data))
        prev_frame = frame

    tb_next: Optional[types.TracebackType] = None
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
    with open(file_path, "rb") as f:
        data = cast(_ExceptionData, pickle.load(f))

    exc = _reconstruct_exc_data(data)

    current_frames: List[types.FrameType] = []
    curr: Optional[types.FrameType] = sys._getframe(1)
    while curr:
        current_frames.append(curr)
        curr = curr.f_back

    if exc.__traceback__ and current_frames:
        reconstructed_outer = exc.__traceback__.tb_frame
        caller_frame = current_frames[0]

        frame_ptr = _PyFrameObject.from_address(id(reconstructed_outer))
        frame_ptr.f_back = id(caller_frame)

    tb_chain: Optional[types.TracebackType] = exc.__traceback__
    for frame in current_frames:
        tb_chain = types.TracebackType(
            tb_next=tb_chain,
            tb_frame=frame,
            tb_lasti=frame.f_lasti,
            tb_lineno=frame.f_lineno,
        )

    exc = exc.with_traceback(tb_chain)
    raise exc
