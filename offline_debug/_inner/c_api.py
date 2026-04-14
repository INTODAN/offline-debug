# Define C API for frame creation
import ctypes
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

_py_decref = ctypes.pythonapi.Py_DecRef
_py_decref.argtypes = (ctypes.py_object,)


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
