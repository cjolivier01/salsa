from __future__ import annotations

import copy
import contextvars
import functools
import itertools
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field, is_dataclass, fields
from enum import IntEnum
from typing import Any, Generic, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

MISSING = object()
_CURRENT_DATABASE: contextvars.ContextVar["Database | None"] = contextvars.ContextVar(
    "pysalsa_current_database",
    default=None,
)


class Durability(IntEnum):
    """How likely an input is to change.

    This first Python port stores durability metadata and revision stamps.
    The validation algorithm is dependency-exact; durability can be used by
    higher-level builders for coarse invalidation policy.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2


class CycleError(RuntimeError):
    """Raised when a tracked query recursively depends on itself."""


@dataclass(frozen=True)
class _DependencyKey:
    kind: str
    parts: tuple[Any, ...]


@dataclass(frozen=True)
class _Dependency:
    key: _DependencyKey
    changed_at: int


@dataclass
class _ActiveQuery:
    key: _DependencyKey
    dependencies: list[_Dependency] = field(default_factory=list)


@dataclass
class _Memo:
    key: _DependencyKey
    tracked: "TrackedFunction[Any, Any]"
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    value: Any
    dependencies: tuple[_Dependency, ...]
    changed_at: int
    verified_at: int


@dataclass
class _InputState:
    value: Any
    changed_at: int
    durability: Durability
    equals: Callable[[Any, Any], bool]
    name: str | None = None


@dataclass
class _ConfigState:
    data: Any
    changed_at: int
    durability: Durability
    equals: Callable[[Any, Any], bool]
    name: str | None = None
    known_paths: set[tuple[Any, ...]] = field(default_factory=set)
    path_changed_at: dict[tuple[Any, ...], int] = field(default_factory=dict)


def equivalent(left: Any, right: Any) -> bool:
    """Best-effort semantic equality for plain Python values.

    Tensor and module-aware equality belongs in :mod:`pysalsa.pytorch`; this
    function deliberately keeps the core package dependency-free.
    """

    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    return False


def _safe_deepcopy(value: T) -> T:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _sort_key(value: Any) -> str:
    return repr(value)


def stable_key(value: Any) -> Hashable:
    """Return a stable-ish hash key for query arguments.

    Hashable values key by value. Common containers key structurally. Other
    objects key by identity, which is the practical choice for handles,
    modules, registries, and user-defined mutable objects.
    """

    if isinstance(value, Input):
        return ("input", id(value._db), value._id)
    if isinstance(value, ConfigInput):
        return ("config", id(value._db), value._id)
    if isinstance(value, tuple):
        return ("tuple", tuple(stable_key(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(stable_key(item) for item in value))
    if isinstance(value, dict):
        items = tuple(
            sorted(
                ((stable_key(key), stable_key(item)) for key, item in value.items()),
                key=_sort_key,
            )
        )
        return ("dict", items)
    if isinstance(value, set):
        return ("set", tuple(sorted((stable_key(item) for item in value), key=_sort_key)))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value),
            tuple((field.name, stable_key(getattr(value, field.name))) for field in fields(value)),
        )
    if isinstance(value, Hashable):
        return ("hash", type(value), value)
    return ("id", type(value), id(value))


def parse_path(path: str | Iterable[Any]) -> tuple[Any, ...]:
    """Parse a dotted config path or normalize an explicit path tuple.

    Use tuples when a real key contains a dot. Numeric dotted segments are
    interpreted as list indexes.
    """

    if isinstance(path, str):
        if path == "":
            return ()
        parsed: list[Any] = []
        for segment in path.split("."):
            parsed.append(int(segment) if segment.isdigit() else segment)
        return tuple(parsed)
    return tuple(path)


def _read_path(data: Any, path: tuple[Any, ...], default: Any = MISSING) -> Any:
    cursor = data
    for part in path:
        try:
            if isinstance(cursor, Mapping):
                cursor = cursor[part]
            else:
                cursor = cursor[part]
        except (KeyError, IndexError, TypeError):
            if default is MISSING:
                dotted = ".".join(str(p) for p in path)
                raise KeyError(dotted) from None
            return default
    return cursor


def _dedupe_dependencies(dependencies: Iterable[_Dependency]) -> tuple[_Dependency, ...]:
    by_key: dict[_DependencyKey, _Dependency] = {}
    for dependency in dependencies:
        by_key[dependency.key] = dependency
    return tuple(by_key.values())


def _ensure_current_database(db: "Database") -> None:
    current = _CURRENT_DATABASE.get()
    if current is not None and current is not db:
        raise ValueError("cannot read an input or config owned by a different Database")


class Input(Generic[T]):
    """A mutable root value stored in a :class:`Database`."""

    __slots__ = ("_db", "_id")

    def __init__(self, db: "Database", input_id: int) -> None:
        self._db = db
        self._id = input_id

    def get(self, *, copy_value: bool = True) -> T:
        _ensure_current_database(self._db)
        state = self._db._inputs[self._id]
        key = _DependencyKey("input", (self._id,))
        self._db._record_dependency(key, state.changed_at)
        return _safe_deepcopy(state.value) if copy_value else state.value

    def set(self, value: T, *, durability: Durability | None = None) -> bool:
        """Set a new value.

        Returns ``True`` when the value changed and the database advanced to a
        new revision. Equivalent values are ignored.
        """

        state = self._db._inputs[self._id]
        if state.equals(state.value, value):
            return False
        if durability is None:
            durability = state.durability
        revision = self._db._advance_revision(durability)
        state.value = _safe_deepcopy(value)
        state.durability = durability
        state.changed_at = revision
        return True

    @property
    def changed_at(self) -> int:
        return self._db._inputs[self._id].changed_at

    def __hash__(self) -> int:
        return hash((id(self._db), self._id))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Input) and self._db is other._db and self._id == other._id

    def __repr__(self) -> str:
        state = self._db._inputs[self._id]
        suffix = f", name={state.name!r}" if state.name else ""
        return f"Input(id={self._id}{suffix})"


class ConfigInput:
    """A YAML/dict-like mutable root with field-level dependency tracking."""

    __slots__ = ("_db", "_id")

    def __init__(self, db: "Database", config_id: int) -> None:
        self._db = db
        self._id = config_id

    def read(self, path: str | Iterable[Any] = (), *, default: Any = MISSING, copy_value: bool = True) -> Any:
        _ensure_current_database(self._db)
        normalized = parse_path(path)
        state = self._db._configs[self._id]
        if normalized not in state.known_paths:
            state.known_paths.add(normalized)
            state.path_changed_at[normalized] = state.changed_at
        changed_at = state.path_changed_at.get(normalized, state.changed_at)
        key = _DependencyKey("config_path", (self._id, normalized))
        self._db._record_dependency(key, changed_at)
        value = _read_path(state.data, normalized, default)
        return _safe_deepcopy(value) if copy_value else value

    def slice(
        self,
        paths: Iterable[str | Iterable[Any]],
        *,
        defaults: Mapping[str | tuple[Any, ...], Any] | None = None,
        copy_value: bool = True,
    ) -> dict[tuple[Any, ...], Any]:
        defaults = defaults or {}
        result: dict[tuple[Any, ...], Any] = {}
        for path in paths:
            normalized = parse_path(path)
            try:
                default = defaults.get(path, defaults.get(normalized, MISSING))
            except TypeError:
                default = defaults.get(normalized, MISSING)
            result[normalized] = self.read(normalized, default=default, copy_value=copy_value)
        return result

    def set(self, data: Any, *, durability: Durability | None = None) -> bool:
        """Replace the config data and mark only changed known paths dirty."""

        state = self._db._configs[self._id]
        new_data = _safe_deepcopy(data)
        if state.equals(state.data, new_data):
            return False
        if durability is None:
            durability = state.durability
        revision = self._db._advance_revision(durability)

        for path in tuple(state.known_paths):
            old_value = _read_path(state.data, path, MISSING)
            new_value = _read_path(new_data, path, MISSING)
            if not state.equals(old_value, new_value):
                state.path_changed_at[path] = revision

        state.data = new_data
        state.changed_at = revision
        state.durability = durability
        return True

    @property
    def changed_at(self) -> int:
        return self._db._configs[self._id].changed_at

    def snapshot(self) -> Any:
        return _safe_deepcopy(self._db._configs[self._id].data)

    def __hash__(self) -> int:
        return hash((id(self._db), self._id))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConfigInput) and self._db is other._db and self._id == other._id

    def __repr__(self) -> str:
        state = self._db._configs[self._id]
        suffix = f", name={state.name!r}" if state.name else ""
        return f"ConfigInput(id={self._id}{suffix})"


class TrackedFunction(Generic[P, R]):
    """Callable wrapper produced by :func:`tracked`."""

    _ids = itertools.count()

    def __init__(
        self,
        fn: Callable[P, R],
        *,
        equals: Callable[[R, R], bool] | None = None,
        reuse_on_equal: bool = True,
        name: str | None = None,
    ) -> None:
        self.fn = fn
        self.equals = equals or equivalent
        self.reuse_on_equal = reuse_on_equal
        self.query_id = next(self._ids)
        self.name = name or f"{fn.__module__}.{fn.__qualname__}"
        functools.update_wrapper(self, fn)

    def __call__(self, db: "Database", *args: P.args, **kwargs: P.kwargs) -> R:
        if not isinstance(db, Database):
            raise TypeError("tracked functions expect a pysalsa.Database as their first argument")
        return db._invoke_tracked(self, args, kwargs)

    def key_for(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> _DependencyKey:
        frozen_kwargs = tuple(sorted((key, stable_key(value)) for key, value in kwargs.items()))
        return _DependencyKey(
            "query",
            (self.query_id, tuple(stable_key(arg) for arg in args), frozen_kwargs),
        )


@overload
def tracked(fn: Callable[P, R], /) -> TrackedFunction[P, R]:
    ...


@overload
def tracked(
    fn: None = None,
    /,
    *,
    equals: Callable[[R, R], bool] | None = None,
    reuse_on_equal: bool = True,
    name: str | None = None,
) -> Callable[[Callable[P, R]], TrackedFunction[P, R]]:
    ...


def tracked(
    fn: Callable[P, R] | None = None,
    /,
    *,
    equals: Callable[[R, R], bool] | None = None,
    reuse_on_equal: bool = True,
    name: str | None = None,
) -> TrackedFunction[P, R] | Callable[[Callable[P, R]], TrackedFunction[P, R]]:
    """Memoize a pure query function.

    The decorated function must take ``Database`` as its first argument.
    """

    def decorate(inner: Callable[P, R]) -> TrackedFunction[P, R]:
        return TrackedFunction(inner, equals=equals, reuse_on_equal=reuse_on_equal, name=name)

    if fn is None:
        return decorate
    return decorate(fn)


class Database:
    """Owns input roots, query memos, and the current revision."""

    def __init__(self) -> None:
        self._revision = 1
        self._input_ids = itertools.count()
        self._config_ids = itertools.count()
        self._inputs: dict[int, _InputState] = {}
        self._configs: dict[int, _ConfigState] = {}
        self._memos: dict[_DependencyKey, _Memo] = {}
        self._active: list[_ActiveQuery] = []
        self._durability_changed_at = {
            Durability.LOW: self._revision,
            Durability.MEDIUM: self._revision,
            Durability.HIGH: self._revision,
        }
        self.query_executions = 0

    @property
    def current_revision(self) -> int:
        return self._revision

    def input(
        self,
        value: T,
        *,
        durability: Durability = Durability.LOW,
        equals: Callable[[T, T], bool] | None = None,
        name: str | None = None,
    ) -> Input[T]:
        input_id = next(self._input_ids)
        self._inputs[input_id] = _InputState(
            value=_safe_deepcopy(value),
            changed_at=self._revision,
            durability=durability,
            equals=equals or equivalent,
            name=name,
        )
        return Input(self, input_id)

    def config(
        self,
        data: Any,
        *,
        durability: Durability = Durability.LOW,
        equals: Callable[[Any, Any], bool] | None = None,
        name: str | None = None,
    ) -> ConfigInput:
        config_id = next(self._config_ids)
        self._configs[config_id] = _ConfigState(
            data=_safe_deepcopy(data),
            changed_at=self._revision,
            durability=durability,
            equals=equals or equivalent,
            name=name,
        )
        return ConfigInput(self, config_id)

    def clear_memos(self) -> None:
        self._memos.clear()

    def memo_count(self) -> int:
        return len(self._memos)

    def last_changed_at(self, durability: Durability) -> int:
        return self._durability_changed_at[durability]

    def _advance_revision(self, durability: Durability) -> int:
        self._revision += 1
        for level in Durability:
            if level <= durability:
                self._durability_changed_at[level] = self._revision
        return self._revision

    def _record_dependency(self, key: _DependencyKey, changed_at: int) -> None:
        if not self._active:
            return
        self._active[-1].dependencies.append(_Dependency(key, changed_at))

    def _invoke_tracked(
        self,
        tracked_fn: TrackedFunction[Any, Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        record_dependency: bool = True,
    ) -> Any:
        key = tracked_fn.key_for(args, kwargs)
        memo = self._memos.get(key)

        if memo is None:
            value = self._execute_query(key, tracked_fn, args, dict(kwargs), None)
        elif memo.verified_at == self._revision or self._memo_is_current(memo):
            value = self._memos[key].value
        else:
            value = self._execute_query(key, tracked_fn, args, dict(kwargs), memo)

        if record_dependency:
            self._record_dependency(key, self._memos[key].changed_at)
        return value

    def _memo_is_current(self, memo: _Memo) -> bool:
        if memo.verified_at == self._revision:
            return True

        for dependency in memo.dependencies:
            if self._dependency_changed_at(dependency.key) != dependency.changed_at:
                return False

        memo.verified_at = self._revision
        return True

    def _dependency_changed_at(self, key: _DependencyKey) -> int:
        if key.kind == "input":
            (input_id,) = key.parts
            return self._inputs[input_id].changed_at
        if key.kind == "config_path":
            config_id, path = key.parts
            state = self._configs[config_id]
            return state.path_changed_at.get(path, state.changed_at)
        if key.kind == "query":
            memo = self._memos.get(key)
            if memo is None:
                return self._revision
            self._invoke_tracked(memo.tracked, memo.args, memo.kwargs, record_dependency=False)
            return self._memos[key].changed_at
        raise KeyError(f"unknown dependency kind: {key.kind}")

    def _execute_query(
        self,
        key: _DependencyKey,
        tracked_fn: TrackedFunction[Any, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        old_memo: _Memo | None,
    ) -> Any:
        if any(active.key == key for active in self._active):
            raise CycleError(f"cycle while executing tracked query {tracked_fn.name}")

        active = _ActiveQuery(key)
        self._active.append(active)
        token = _CURRENT_DATABASE.set(self)
        self.query_executions += 1
        try:
            new_value = tracked_fn.fn(self, *args, **kwargs)
        finally:
            _CURRENT_DATABASE.reset(token)
            popped = self._active.pop()
            assert popped is active

        dependencies = _dedupe_dependencies(active.dependencies)
        if old_memo is not None and tracked_fn.equals(old_memo.value, new_value):
            value = old_memo.value if tracked_fn.reuse_on_equal else new_value
            changed_at = old_memo.changed_at
        else:
            value = new_value
            changed_at = self._revision

        self._memos[key] = _Memo(
            key=key,
            tracked=tracked_fn,
            args=args,
            kwargs=kwargs,
            value=value,
            dependencies=dependencies,
            changed_at=changed_at,
            verified_at=self._revision,
        )
        return value
