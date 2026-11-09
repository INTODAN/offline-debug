"""Tests for the load_traceback module."""

from __future__ import annotations

import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pytest

from offline_debug import load_traceback, save_traceback

if TYPE_CHECKING:
    import types


class Vanished:
    """Class removed from this module before loading, like one that moved between versions."""

    def __init__(self, value: int) -> None:
        """Keep ``value`` as both a constructor argument and pickled state."""
        self.value = value

    def __reduce__(self) -> tuple:
        """Pickle as a call with arguments followed by state."""
        return Vanished, (self.value,), {"value": self.value}


class VanishedError(Exception):
    """Exception class removed from this module before loading."""


class VanishedDict(dict):
    """Dict subclass, whose items pickle sets one by one on the rebuilt object."""


class VanishedList(list):
    """List subclass, whose items pickle appends to the rebuilt object."""


class VanishedKeywords:
    """Class rebuilt by calling ``__new__`` with keyword arguments."""

    size: int

    def __new__(cls, *, size: int) -> Self:
        """Require ``size`` by keyword."""
        self = super().__new__(cls)
        self.size = size
        return self

    def __getnewargs_ex__(self) -> tuple[tuple, dict[str, int]]:
        """Pickle the keyword argument ``__new__`` requires."""
        return (), {"size": self.size}


def get_frames(tb: types.TracebackType | None) -> list[types.FrameType]:
    """Extract all frames from a traceback."""
    frames = []
    curr = tb
    while curr:
        frames.append(curr.tb_frame)
        curr = curr.tb_next
    return frames


def test_load_invalid_object(tmp_path: Path) -> None:
    """Test that load_traceback raises TypeError when loading an invalid object."""
    import pickle

    dump_file = tmp_path / "invalid.dump"
    with dump_file.open("wb") as f:
        pickle.dump("not an ExceptionData object", f)

    with pytest.raises(TypeError, match="Expected _ExceptionData, but got str"):
        load_traceback(dump_file)


def test_load_non_existent_file() -> None:
    """Test that load_traceback raises FileNotFoundError when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_traceback(Path("non_existent_file.dump"))


def test_reconstruct_invalid_exception_type() -> None:
    """Test that _reconstruct_exc_data raises TypeError when the pickled exception is invalid."""
    import pickle

    from offline_debug._inner.load_traceback import _reconstruct_exc_data
    from offline_debug._inner.models import ExceptionData

    data = ExceptionData(
        exc_pickle=pickle.dumps("not an exception"),
        tb_frames=[],
    )

    with pytest.raises(TypeError, match="Expected BaseException, but got str"):
        _reconstruct_exc_data(data)


def test_reconstruct_invalid_frame_type(monkeypatch) -> None:
    """Test that _reconstruct_exc_data raises TypeError when frame creation fails."""
    import offline_debug._inner.c_api._create_frame as _create_frame_module
    from offline_debug._inner.load_traceback import _reconstruct_exc_data
    from offline_debug._inner.models import CodeData, ExceptionData, FrameData

    # Mock _get_py_frame_new to return a function that returns something that is not a FrameType
    monkeypatch.setattr(_create_frame_module, "_get_py_frame_new", lambda: lambda *_: "not a frame")

    import pickle

    data = ExceptionData(
        exc_pickle=pickle.dumps(ValueError("test")),
        tb_frames=[
            FrameData(
                code=CodeData(filename="dummy.py", name="dummy", qualname="dummy", firstlineno=1),
                globals={},
                locals={},
                lasti=0,
                lineno=0,
                stack_depth=0,
            )
        ],
    )

    with pytest.raises(TypeError, match=r"Expected types.FrameType, but got str"):
        _reconstruct_exc_data(data)


def test_load_traceback_should_raise_false() -> None:
    """Test load_traceback when should_raise is False."""
    buffer = BytesIO()
    exc = ValueError("test_raise_false")
    save_traceback(exc, buffer)
    buffer.seek(0)

    loaded_exc = load_traceback(buffer, should_raise=False)
    assert isinstance(loaded_exc, ValueError)
    assert str(loaded_exc) == "test_raise_false"


def test_loaded_traceback_is_formattable() -> None:
    """The ``traceback`` module must format reconstructed frames from ``tb_lineno``."""
    buffer = BytesIO()

    def raise_late() -> None:
        msg = "late"
        raise ValueError(msg)

    try:
        raise_late()
    except ValueError as e:
        save_traceback(e, buffer)
    buffer.seek(0)

    loaded_exc = load_traceback(buffer, should_raise=False)
    formatted = "".join(traceback.format_exception(loaded_exc))
    assert "in raise_late\n    raise ValueError(msg)\n" in formatted
    assert formatted.endswith("ValueError: late\n")


def test_missing_class_loads_as_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value whose class cannot be imported must not make the whole dump unloadable."""
    buffer = BytesIO()

    def raise_with_vanished() -> None:
        _vanished = Vanished(1)
        msg = "vanished"
        raise ValueError(msg)

    try:
        raise_with_vanished()
    except ValueError as e:
        save_traceback(e, buffer)
    buffer.seek(0)
    monkeypatch.delattr(sys.modules[__name__], "Vanished")

    loaded_exc = load_traceback(buffer, should_raise=False)
    stand_in = get_frames(loaded_exc.__traceback__)[-1].f_locals["_vanished"]
    assert repr(stand_in) == f"<missing {__name__}.Vanished(1,)>"
    assert stand_in.value == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (VanishedDict(a=1), "<missing {module}.VanishedDict()> with items {{'a': 1}}"),
        (VanishedList([1, 2]), "<missing {module}.VanishedList()> with items [1, 2]"),
        (
            VanishedKeywords(size=3),
            "<missing {module}.VanishedKeywords(){{'size': 3}}> with items None",
        ),
    ],
)
def test_missing_class_stands_in_however_pickle_rebuilds_it(
    monkeypatch: pytest.MonkeyPatch, value: object, expected: str
) -> None:
    """Items of list and dict subclasses and keyword arguments are kept by the stand-in."""
    buffer = BytesIO()

    def raise_with_value() -> None:
        _value = value
        msg = "vanished"
        raise ValueError(msg)

    try:
        raise_with_value()
    except ValueError as e:
        save_traceback(e, buffer)
    buffer.seek(0)
    monkeypatch.delattr(sys.modules[__name__], type(value).__name__)

    loaded_exc = load_traceback(buffer, should_raise=False)
    stand_in = get_frames(loaded_exc.__traceback__)[-1].f_locals["_value"]
    described = f"{stand_in!r} with items {getattr(stand_in, 'items', None)}"
    assert described == expected.format(module=__name__)


def test_missing_exception_class_loads_as_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception whose own class cannot be imported still loads, with its frames."""
    buffer = BytesIO()

    def raise_vanished() -> None:
        _detail = "kept"
        msg = "gone"
        raise VanishedError(msg)

    try:
        raise_vanished()
    except VanishedError as e:
        save_traceback(e, buffer)
    buffer.seek(0)
    monkeypatch.delattr(sys.modules[__name__], "VanishedError")

    loaded_exc = load_traceback(buffer, should_raise=False)
    assert type(loaded_exc) is RuntimeError
    assert str(loaded_exc) == f"Unloadable exception <missing {__name__}.VanishedError('gone',)>"
    assert get_frames(loaded_exc.__traceback__)[-1].f_locals["_detail"] == "kept"


def test_dump_with_marshalled_code_still_loads() -> None:
    """A dump written before ``CodeData``, with marshalled code objects, stays loadable."""
    import marshal
    import pickle

    from offline_debug import CodeData, parse_traceback

    def raise_old() -> None:
        msg = "old"
        raise ValueError(msg)

    try:
        raise_old()
    except ValueError as e:
        data = save_traceback(e, None)
    old_code: Any = marshal.dumps(raise_old.__code__)
    data.tb_frames[-1].code = old_code
    buffer = BytesIO(pickle.dumps(data))

    parsed = parse_traceback(buffer)
    assert parsed.tb_frames[-1].code == CodeData(
        filename=__file__,
        name="raise_old",
        qualname=raise_old.__qualname__,
        firstlineno=raise_old.__code__.co_firstlineno,
    )
    buffer.seek(0)
    loaded_exc = load_traceback(buffer, should_raise=False)
    assert get_frames(loaded_exc.__traceback__)[-1].f_code.co_name == "raise_old"
