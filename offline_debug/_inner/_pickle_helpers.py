"""
Pickling that never aborts a dump because of one object.

Objects that vanilla pickle rejects are replaced by a placeholder string, at
any depth, so a single lock, socket or connection nested inside a value costs
only that leaf and not the whole value. The failure points of pickling are
leaves: an object's ``__reduce_ex__`` raising, or a class or function that
cannot be found by its qualified name. Both are checked on the way down, once
per object, so the dump stays a single pass with native speed for containers.

``BaseException`` reduces to ``(type(self), self.args, state)``, so pickle
rebuilds an exception by calling ``Cls(*self.args)`` and then applying state.
Exceptions whose ``__init__``/``__new__`` takes required keyword-only arguments
(and which do not populate ``self.args`` via ``super().__init__``) pickle fine
but fail on load with a ``TypeError``. Since a traceback-capture library must
serialize whatever the user raised, we take over reconstruction for such
exceptions.

We keep the takeover as narrow as possible so well-behaved exceptions keep
their vanilla pickle behavior:

- Classes that customize the pickle protocol (``__reduce__``, ``__setstate__``,
  ...) are left alone; their hooks know how to rebuild them (e.g. ``OSError``).
- Classes whose constructors are all C-level (every builtin exception) are left
  alone; ``Cls(*args)`` is guaranteed to work and restores C-level state such
  as ``StopIteration.value`` that lives outside ``__dict__``.
- Only classes with a Python-defined ``__init__``/``__new__`` — the only kind
  that can demand required keyword-only arguments — are reconstructed by us,
  and even then reconstruction first retries normal construction and only
  falls back to ``__new__`` (bypassing the constructor) when it fails.

Exception groups get a dedicated reconstructor because a bare
``cls.__new__(cls)`` is not enough for them: ``BaseExceptionGroup.__new__``
requires ``message``/``exceptions`` to build a valid group for any subclass.

.. warning::
    The path of this module and the names of :func:`reconstruct_exception` and
    :func:`reconstruct_exception_group` are embedded in every dump that takes
    the takeover path. Renaming or moving them makes previously saved dumps
    unloadable, so treat them as part of the on-disk format.
"""

import contextlib
import copyreg
import io
import pickle
import types
from collections.abc import Callable, Iterable
from typing import IO, Any, Self

# Pickle protocol hooks that let a class control its own serialization.
# If an exception class customizes any of these, we must not override its
# reduction: the class (e.g. OSError, ImportError) knows how to rebuild itself.
_PICKLE_PROTOCOL_HOOKS = (
    "__reduce_ex__",
    "__reduce__",
    "__getstate__",
    "__setstate__",
    "__getnewargs__",
    "__getnewargs_ex__",
)


def _customizes_pickling(cls: type[BaseException]) -> bool:
    """Whether ``cls`` overrides any pickle protocol hook of ``BaseException``."""
    return any(
        getattr(cls, hook, None) is not getattr(BaseException, hook, None)
        for hook in _PICKLE_PROTOCOL_HOOKS
    )


def _has_python_constructor(cls: type[BaseException]) -> bool:
    """
    Whether ``cls`` has a Python-defined ``__init__`` or ``__new__``.

    Builtin exceptions expose C slot wrappers here, and for them default pickle
    reconstruction always works. Only a Python-level constructor can introduce
    required keyword-only arguments that break ``Cls(*args)`` at load time.
    """
    return isinstance(cls.__init__, types.FunctionType) or isinstance(
        cls.__new__, types.FunctionType
    )


def _takes_over(cls: type[BaseException]) -> bool:
    """
    Whether we reconstruct ``cls`` ourselves instead of leaving it to pickle.

    Default reduction is guaranteed to round-trip for the other classes, and
    taking over would lose state the class restores itself (e.g. ``OSError.errno``).
    """
    return not _customizes_pickling(cls) and _has_python_constructor(cls)


# Decides, from a value's type alone, whether it is a proxy the save must not touch.
ProxyMatcher = Callable[[object], bool]


def _no_proxies(value: object) -> bool:  # noqa: ARG001 - the matcher for an empty registry
    return False


def proxy_matcher(entries: Iterable[type | str] = ()) -> ProxyMatcher:
    """
    Build the predicate that recognises the proxies a save must never touch.

    A proxy such as an ``rpyc`` netref turns every instance operation - attribute
    reads, ``repr``, ``hash``, ``isinstance`` through ``__class__``, pickling
    through ``__reduce_ex__`` - into a remote call that blocks for the peer's full
    timeout on a broken connection. The save only wants to note that the value was
    there.

    An entry is a class, or the fully-qualified name of one such as
    ``"rpyc.core.netref.BaseNetref"``, which names a proxy family without importing
    its package. Either form matches subclasses, since only ``type(value).__mro__``
    is consulted - the one thing guaranteed local for every object, and a property
    the predicate must keep: it runs before the pickler's own ``isinstance``.
    """
    classes: set[type] = set()
    names: set[str] = set()
    for entry in entries:
        if isinstance(entry, type):
            classes.add(entry)
        elif isinstance(entry, str):
            names.add(entry)
        else:
            msg = (
                "proxy_types entries must be classes or fully-qualified class names, "
                f"got {type(entry).__name__}"
            )
            raise TypeError(msg)
    if not classes and not names:
        return _no_proxies

    def matches(value: object) -> bool:
        return any(
            cls in classes or f"{cls.__module__}.{cls.__qualname__}" in names
            for cls in type(value).__mro__
        )

    return matches


def proxy_placeholder(value: object) -> str:
    """Describe a proxy from local facts only: its class and its identity."""
    cls = type(value)
    return f"<proxy {cls.__module__}.{cls.__qualname__} at 0x{id(value):x}>"


def reconstruct_exception(
    cls: type[BaseException], args: tuple[Any, ...], state: dict[str, Any] | None
) -> BaseException:
    """Rebuild an exception, bypassing its constructor only if it rejects ``args``."""
    try:
        exc = cls(*args)
    except Exception:  # noqa: BLE001 - a rejecting constructor is exactly the case we handle
        exc = BaseException.__new__(cls)
        # Suppressed failures happen when e.g. `args` is shadowed by a read-only property.
        with contextlib.suppress(Exception):
            exc.args = args
    if state:
        exc.__dict__.update(state)
    return exc


def reconstruct_exception_group(
    cls: type[BaseExceptionGroup[Any]],
    message: str,
    exceptions: tuple[BaseException, ...],
    state: dict[str, Any] | None,
) -> BaseExceptionGroup[Any]:
    """
    Rebuild an exception group, bypassing its constructor only when necessary.

    ``BaseExceptionGroup.__new__`` establishes ``message``/``exceptions`` for any
    subclass, so when normal construction fails (e.g. a subclass demanding extra
    required keyword-only arguments) we call it directly.
    """
    try:
        exc = cls(message, exceptions)
    except Exception:  # noqa: BLE001 - a rejecting constructor is exactly the case we handle
        exc = BaseExceptionGroup.__new__(cls, message, exceptions)
    if state:
        exc.__dict__.update(state)
    return exc


def safe_repr(value: object) -> str:
    """
    ``repr(value)``, falling back to the bare object repr if it raises.

    A placeholder describes a value that already failed to pickle, whose ``repr``
    may well fail too - and a placeholder that raises would lose the whole dump.
    """
    try:
        return repr(value)
    except BaseException:  # noqa: BLE001 - see docstring
        return object.__repr__(value)


def placeholder(obj: object) -> str:
    """Stand-in text for an object a dump cannot carry."""
    return f"<unpicklable {type(obj).__name__}: {safe_repr(obj)}>"


class ResilientPickler(pickle.Pickler):
    """
    Pickler that replaces what it cannot pickle with placeholders, at any depth.

    It also takes over reconstruction of constructor-rejecting exceptions, and
    writes a placeholder for a registered proxy (see :func:`proxy_matcher`)
    wherever one appears - nested in a container, in an exception's ``args`` or
    attributes - instead of asking it how to pickle itself, which is a remote call.
    """

    def __init__(self, file: IO[bytes], is_proxy: ProxyMatcher = _no_proxies) -> None:
        """Pickle into ``file``, remembering per dump which classes pickle by name."""
        super().__init__(file)
        self._is_proxy = is_proxy
        # Whether anything was written as a placeholder instead of itself.
        self.substituted = False
        # Keyed by id, so each entry also holds its object: a reconstructor can be a
        # temporary (a bound method built by ``__reduce__``), and a freed one's id
        # is soon reused by another object that would inherit its answer.
        self._by_reference: dict[int, tuple[object, bool]] = {}

    def _pickles_by_reference(self, obj: object) -> bool:
        """Whether a class or function is found under its qualified name (cached per dump)."""
        cached = self._by_reference.get(id(obj))
        if cached is None:
            try:
                pickle.dumps(obj)
                by_reference = True
            except Exception:  # noqa: BLE001 - not being picklable is the case we handle
                by_reference = False
            cached = self._by_reference[id(obj)] = (obj, by_reference)
        return cached[1]

    def _stand_in(self, text: str) -> tuple[type[str], tuple[str]]:
        """Reduce an object to placeholder ``text``, rebuilt on load as a plain string."""
        self.substituted = True
        return str, (text,)

    def _reduction(self, obj: object) -> object:
        """
        Return what to write for ``obj``, or ``None`` when a load could not rebuild it.

        ``NotImplemented`` leaves the object to the pickler's own handling.
        """
        if isinstance(obj, (type, types.FunctionType)):
            # Pickled by name; leave the pickler's own handling in place when it works.
            return NotImplemented if self._pickles_by_reference(obj) else None
        if isinstance(obj, BaseException) and _takes_over(type(obj)):
            reduced = _exception_reduce(obj)
        else:
            # Mirror the pickler's own lookup order so the reduction is identical
            # to what it would compute, without computing it twice.
            reduce = copyreg.dispatch_table.get(type(obj))
            reduced = reduce(obj) if reduce else obj.__reduce_ex__(pickle.DEFAULT_PROTOCOL)
        if isinstance(reduced, tuple) and not self._pickles_by_reference(_reconstructor(reduced)):
            # Loading would import the reconstructor by name and fail there.
            return None
        return reduced

    # The inline suppression below is needed because this returns NotImplemented to
    # fall back to default reduction, which the stdlib stub's return type omits.
    def reducer_override(self, obj: object, /) -> object:  # ty: ignore[invalid-method-override]
        # Proxies first: even an isinstance() can reach the peer through a proxy's
        # __class__, so nothing may touch the instance before this.
        if self._is_proxy(obj):
            return self._stand_in(proxy_placeholder(obj))
        try:
            reduced = self._reduction(obj)
        except Exception:  # noqa: BLE001 - not being picklable is the case we handle
            reduced = None
        return self._stand_in(placeholder(obj)) if reduced is None else reduced


def _reconstructor(reduced: tuple[Any, ...]) -> object:
    """Return the callable a load must import to rebuild an object from its reduction."""
    callable_, args = reduced[0], reduced[1]
    # pickle itself recognizes copyreg's __newobj__/__newobj_ex__ by name, and our
    # own reconstructors are importable: for all of them the class is what can be missing.
    names_a_class = (
        callable_ is reconstruct_exception
        or callable_ is reconstruct_exception_group
        or getattr(callable_, "__name__", None) in ("__newobj__", "__newobj_ex__")
    )
    return args[0] if names_a_class else callable_


def _exception_reduce(exc: BaseException) -> tuple[Any, tuple[Any, ...]]:
    """Take over reconstruction of an exception whose constructor may reject ``args``."""
    state = exc.__dict__.copy() or None
    # Groups need their dedicated reconstructor: message/exceptions live
    # behind __new__ rather than a settable `args`.
    if isinstance(exc, BaseExceptionGroup):
        return reconstruct_exception_group, (type(exc), exc.message, exc.exceptions, state)
    return reconstruct_exception, (type(exc), exc.args, state)


def resilient_dump(obj: object, file: IO[bytes], is_proxy: ProxyMatcher = _no_proxies) -> bool:
    """
    Serialize ``obj`` to an open binary ``file`` using :class:`ResilientPickler`.

    Returns whether anything inside ``obj`` was written as a placeholder.
    """
    pickler = ResilientPickler(file, is_proxy)
    pickler.dump(obj)
    return pickler.substituted


def resilient_dumps(obj: object, is_proxy: ProxyMatcher = _no_proxies) -> bytes:
    """Serialize ``obj`` to bytes using :class:`ResilientPickler`."""
    buf = io.BytesIO()
    resilient_dump(obj, buf, is_proxy)
    return buf.getvalue()


class MissingObject:
    """
    Stand-in for a pickled object whose class or reconstructor cannot be imported here.

    :class:`TolerantUnpickler` creates a subclass per missing ``module.name`` and
    the dump uses it however it used the original: calling it keeps the arguments,
    and any state pickled afterwards becomes attributes.
    """

    args: tuple[Any, ...]

    def __new__(cls, *args: Any) -> Self:  # noqa: ANN401
        """Keep the arguments the dump called the missing name with."""
        self = super().__new__(cls)
        self.args = args
        return self

    def __setstate__(self, state: object) -> None:
        """Apply pickled state as attributes; keep non-dict state (e.g. slots) as one."""
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})

    def __repr__(self) -> str:
        """Show which name is missing and how the dump called it."""
        return f"<missing {type(self).__module__}.{type(self).__qualname__}{self.args!r}>"


class TolerantUnpickler(pickle.Unpickler):
    """Unpickler that stands in :class:`MissingObject` for names it cannot import."""

    def find_class(self, module: str, name: str, /) -> Any:  # noqa: ANN401
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (MissingObject,), {"__module__": module, "__qualname__": name})


def tolerant_load(file: IO[bytes]) -> Any:  # noqa: ANN401
    """Deserialize from an open binary ``file`` using :class:`TolerantUnpickler`."""
    return TolerantUnpickler(file).load()


def tolerant_loads(data: bytes) -> Any:  # noqa: ANN401
    """Deserialize ``data`` using :class:`TolerantUnpickler`."""
    return tolerant_load(io.BytesIO(data))
