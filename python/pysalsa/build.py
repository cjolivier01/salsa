from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .core import MISSING, ConfigInput, Database, equivalent, tracked


@dataclass(frozen=True)
class ComponentResult:
    name: str
    key: tuple[Any, ...]
    spec_version: int
    value: Any
    equals: Callable[[Any, Any], bool]
    reuse_on_equal: bool


def _component_results_equivalent(left: ComponentResult, right: ComponentResult) -> bool:
    if (
        left.name != right.name
        or left.key != right.key
        or left.spec_version != right.spec_version
        or not right.reuse_on_equal
    ):
        return False
    return right.equals(left.value, right.value)


@dataclass
class ComponentSpec:
    name: str
    builder: Callable[..., Any]
    equals: Callable[[Any, Any], bool]
    reuse_on_equal: bool
    version: int


class ComponentContext:
    """Context object passed to :class:`ComponentGraph` builders."""

    def __init__(self, db: Database, graph: "ComponentGraph", config: ConfigInput, key: tuple[Any, ...]) -> None:
        self.db = db
        self.graph = graph
        self.config = config
        self.key = key

    def read(self, path, *, default=MISSING, copy_value: bool = True):
        return self.config.read(path, default=default, copy_value=copy_value)

    def component(self, name: str, *key: Any) -> Any:
        return self.graph.build(self.db, self.config, name, *key)


class ComponentGraph:
    """Small helper for declarative config/component build graphs.

    Builders still use normal tracked-query semantics; this class mainly gives
    names, dynamic keys, and per-component equality policies.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ComponentSpec] = {}
        self._versions: dict[str, int] = {}
        self._frozen = False
        self._executions: list[tuple[str, tuple[Any, ...]]] = []

    def component(
        self,
        name: str,
        *,
        equals: Callable[[Any, Any], bool] | None = None,
        reuse_on_equal: bool = True,
    ):
        def decorate(builder: Callable[..., Any]):
            self.register(name, builder, equals=equals, reuse_on_equal=reuse_on_equal)
            return builder

        return decorate

    def register(
        self,
        name: str,
        builder: Callable[..., Any],
        *,
        equals: Callable[[Any, Any], bool] | None = None,
        reuse_on_equal: bool = True,
    ) -> None:
        if self._frozen:
            raise RuntimeError("cannot register components after a ComponentGraph has been used")
        version = self._versions.get(name, 0) + 1
        self._versions[name] = version
        self._specs[name] = ComponentSpec(
            name=name,
            builder=builder,
            equals=equals or equivalent,
            reuse_on_equal=reuse_on_equal,
            version=version,
        )

    def build(self, db: Database, config: ConfigInput, name: str, *key: Any) -> Any:
        return self.build_result(db, config, name, *key).value

    def build_result(self, db: Database, config: ConfigInput, name: str, *key: Any) -> ComponentResult:
        self._frozen = True
        return _build_component(db, self, config, name, tuple(key))

    def clear_executions(self) -> None:
        self._executions.clear()

    def executions(self) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        return tuple(self._executions)


@tracked(equals=_component_results_equivalent, reuse_on_equal=True)
def _build_component(
    db: Database,
    graph: ComponentGraph,
    config: ConfigInput,
    name: str,
    key: tuple[Any, ...],
) -> ComponentResult:
    try:
        spec = graph._specs[name]
    except KeyError as exc:
        raise KeyError(f"unknown component {name!r}") from exc
    graph._executions.append((name, key))
    ctx = ComponentContext(db, graph, config, key)
    value = spec.builder(ctx, *key)
    return ComponentResult(
        name=name,
        key=key,
        spec_version=spec.version,
        value=value,
        equals=spec.equals,
        reuse_on_equal=spec.reuse_on_equal,
    )
