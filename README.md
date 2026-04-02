# Traceback Serializer Project (`offline-debug`)

[![PyPI version](https://img.shields.io/pypi/v/offline-debug.svg)](https://pypi.org/project/offline-debug/)
[![Tests](https://github.com/INTODAN/offline-debug/actions/workflows/ci.yml/badge.svg)](https://github.com/INTODAN/offline-debug/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/INTODAN/offline-debug)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ty checked](https://img.shields.io/badge/ty-checked-blue.svg)](https://github.com/intodan/ty)

## Overview
A Python package for high-fidelity serialization and deserialization of exceptions and their complete tracebacks. Unlike other solutions, `offline-debug` reconstructs **actual** `types.FrameType` objects using the Python C API, ensuring that re-raised exceptions look and feel genuine to debuggers and introspection tools.

## Core Functions
- `save_traceback(exc: Exception, file_path: str)`: 
  Serializes an exception, its traceback, and all picklable local/global variables to a binary file.
- `load_traceback(file_path: str) -> typing.Never`: 
  Loads the serialized state, reconstructs the exception and its full traceback chain (including `__cause__` and `__context__`), and raises it.

## Technical Implementation
- **True Frame Reconstruction**: Uses `ctypes` to call `PyFrame_New` from the Python C API. This creates real `frame` objects which are required for a valid `types.TracebackType`.
- **Python 3.13 Compatibility**: Leverages PEP 667 features where `f_locals` is a write-through proxy, allowing for accurate local variable restoration.
- **Robust Serialization**: 
  - `pickle` is used for exceptions and variables.
  - `marshal` is used for code objects.
  - Non-picklable items are gracefully handled by storing their `repr`.

## Development & Tooling
- **Package Manager**: `uv`
- **Minimum Python**: 3.12
- **Testing**: `pytest`
- **Commands**:
  - Add dependencies: `uv add <package>`
  - Run tests: `uv run pytest`

## Usage Example
```python
from offline_debug import save_traceback, load_traceback

try:
    # Code that might fail
    some_complex_operation()
except Exception as e:
    save_traceback(e, "crash_report.dump")

# To debug or re-examine later:
load_traceback("crash_report.dump")
```
