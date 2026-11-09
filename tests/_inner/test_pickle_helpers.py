"""Tests for the resilient pickler."""

from __future__ import annotations

import pickle
import threading
from typing import Any, Never

from offline_debug._inner._pickle_helpers import resilient_dumps


class Job:
    """Object holding an unpicklable lock next to plain state."""

    def __init__(self) -> None:
        """Mix picklable and unpicklable attributes."""
        self.name = "nightly"
        self.lock = threading.Lock()

    def describe(self) -> str:
        """Bound methods pickle through ``getattr`` on their instance."""
        return self.name


class Unreducible:
    """Object whose own pickle hook rejects it."""

    def __reduce__(self) -> tuple:
        """Refuse to be pickled."""
        msg = "cannot pickle me"
        raise TypeError(msg)


class Hostile:
    """Object that fails on every access, even the ``__class__`` read of ``isinstance``."""

    __slots__ = ()

    def __getattribute__(self, name: str) -> Never:
        """Refuse every attribute."""
        raise AssertionError(name)


class Rebuilt:
    """Object rebuilt on load by a bound method, a reconstructor created per reduction."""

    def __init__(self, owner: Owner) -> None:
        """Remember whose ``rebuild`` method reconstructs this object."""
        self.owner = owner

    def __reduce__(self) -> tuple:
        """Reduce to a fresh bound method of the owner."""
        return self.owner.rebuild, ()


class Owner:
    """Importable owner whose bound method serves as a reconstructor."""

    def rebuild(self) -> str:
        """Stand in for the rebuilt object."""
        return "rebuilt"


class LockedOwner(Owner):
    """Owner that cannot be pickled, so neither can its bound methods."""

    def __init__(self) -> None:
        """Hold a lock."""
        self.lock = threading.Lock()


def roundtrip(obj: object) -> Any:  # noqa: ANN401
    """Dump ``obj`` with the resilient pickler and load it back with vanilla pickle."""
    return pickle.loads(resilient_dumps(obj))  # noqa: S301


def test_nested_unpicklable_values_become_placeholders() -> None:
    """Only the unpicklable leaf is replaced, whatever its depth."""
    loaded = roundtrip({"job": Job(), "items": [1, Unreducible()]})

    assert loaded["job"].name == "nightly"
    assert loaded["job"].lock.startswith("<unpicklable lock: ")
    assert loaded["items"][0] == 1
    assert loaded["items"][1].startswith("<unpicklable Unreducible: ")


def test_unresolvable_names_become_placeholders() -> None:
    """Classes, functions and instances whose class has no importable name are replaced."""

    class Local:
        pass

    loaded = roundtrip([Local(), Local, lambda: None, Job])

    assert loaded[0].startswith("<unpicklable Local: ")
    assert loaded[1].startswith("<unpicklable type: ")
    assert loaded[2].startswith("<unpicklable function: ")
    assert loaded[3] is Job


def test_callables_pickled_through_getattr_are_kept() -> None:
    """Builtins, descriptors and bound methods have no importable type but reduce fine."""
    loaded = roundtrip([len, str.upper, Job().describe])

    assert loaded[0] is len
    assert loaded[1] is str.upper
    assert loaded[2]() == "nightly"


def test_copyreg_reducers_still_apply() -> None:
    """Types pickled through ``copyreg.dispatch_table`` keep their reducer."""
    number = 1j
    assert roundtrip(number) == number


def test_object_failing_on_every_access_costs_only_itself() -> None:
    """A value that raises as soon as it is inspected is replaced without its container."""
    loaded = roundtrip({"keep": 1, "bad": Hostile()})

    assert loaded["keep"] == 1
    assert loaded["bad"].startswith("<unpicklable Hostile: <")


def test_local_exception_class_costs_only_the_exception() -> None:
    """An exception we reconstruct ourselves is replaced when its class has no importable name."""

    class LocalError(Exception):
        def __init__(self, *, code: int) -> None:
            super().__init__()
            self.code = code

    loaded = roundtrip([1, LocalError(code=2)])

    assert loaded[0] == 1
    assert loaded[1].startswith("<unpicklable LocalError: ")


def test_reconstructor_verdict_is_not_inherited_by_address() -> None:
    """A freed reconstructor's verdict must not apply to the next object at its address."""
    locked, owner = LockedOwner(), Owner()

    # Each reduction builds a bound method that is freed straight away, so one for
    # ``owner`` soon lands on the address of an earlier one for ``locked``.
    loaded = roundtrip([Rebuilt(locked if i % 2 else owner) for i in range(100)])

    assert all(value.startswith("<unpicklable Rebuilt: ") for value in loaded[1::2])
    assert loaded[::2] == ["rebuilt"] * 50
