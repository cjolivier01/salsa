from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .build import ComponentGraph
from .core import ConfigInput, Database


@dataclass(frozen=True)
class ComponentTarget:
    """A named component and dynamic key to build through `ComponentGraph`."""

    name: str
    key: tuple[Any, ...] = ()

    def label(self) -> str:
        if not self.key:
            return self.name
        suffix = ",".join(str(part) for part in self.key)
        return f"{self.name}:{suffix}"


def target(name: str, *key: Any) -> ComponentTarget:
    normalized_key = tuple(key)
    hash(normalized_key)
    return ComponentTarget(name, normalized_key)


def _normalize_target(value: ComponentTarget | str | tuple[Any, ...]) -> ComponentTarget:
    if isinstance(value, ComponentTarget):
        return value
    if isinstance(value, str):
        return ComponentTarget(value)
    if not value:
        raise ValueError("component target tuples must contain at least a name")
    name, *key = value
    if not isinstance(name, str):
        raise TypeError("component target tuple names must be strings")
    normalized_key = tuple(key)
    hash(normalized_key)
    return ComponentTarget(name, normalized_key)


def _target_sort_key(component: ComponentTarget) -> tuple[str, str]:
    return (component.name, repr(component.key))


def _sort_targets(targets: Iterable[ComponentTarget]) -> tuple[ComponentTarget, ...]:
    return tuple(sorted(targets, key=_target_sort_key))


@dataclass(frozen=True)
class ComponentSnapshot:
    values: Mapping[ComponentTarget, Any]

    def labels(self) -> list[str]:
        return sorted(component.label() for component in self.values)


@dataclass(frozen=True)
class RebuildReport:
    added: tuple[ComponentTarget, ...]
    removed: tuple[ComponentTarget, ...]
    rebuilt: tuple[ComponentTarget, ...]
    reused: tuple[ComponentTarget, ...]
    executed: tuple[ComponentTarget, ...]
    executed_targets: tuple[ComponentTarget, ...]

    @property
    def rebuild_set(self) -> tuple[ComponentTarget, ...]:
        return _sort_targets((*self.added, *self.rebuilt))

    def labels(self, targets: Iterable[ComponentTarget]) -> list[str]:
        return sorted(target.label() for target in targets)

    def to_golden(self) -> dict[str, list[str]]:
        return {
            "added": self.labels(self.added),
            "rebuilt": self.labels(self.rebuilt),
            "removed": self.labels(self.removed),
            "reused": self.labels(self.reused),
            "executed": self.labels(self.executed),
            "executed_targets": self.labels(self.executed_targets),
            "rebuild_set": self.labels(self.rebuild_set),
        }


TargetSelector = Callable[[Any], Iterable[ComponentTarget | str | tuple[Any, ...]]]


class RebuildPlanner:
    """Compare component identity across config revisions.

    The planner deliberately reports identity preservation, not semantic
    equality. Component builders can use `ComponentGraph` equality/backdating to
    keep an old object when a recomputation proves the result equivalent.
    """

    def __init__(
        self,
        graph: ComponentGraph,
        config: Any,
        targets: TargetSelector,
        *,
        db: Database | None = None,
        config_input: ConfigInput | None = None,
    ) -> None:
        self.graph = graph
        if config_input is not None:
            if db is not None and config_input._db is not db:
                raise ValueError("config_input must be owned by the provided Database")
            self.db = config_input._db
            self.config = config_input
            self.config.set(config)
        else:
            self.db = db or Database()
            self.config = self.db.config(config)
        self._select_targets = targets
        self.snapshot = self._build_snapshot(config)

    def _targets_for(self, config_data: Any) -> tuple[ComponentTarget, ...]:
        seen: set[ComponentTarget] = set()
        result: list[ComponentTarget] = []
        for item in self._select_targets(config_data):
            component = _normalize_target(item)
            if component not in seen:
                seen.add(component)
                result.append(component)
        return tuple(result)

    def _build_snapshot(self, config_data: Any) -> ComponentSnapshot:
        values: dict[ComponentTarget, Any] = {}
        for component in self._targets_for(config_data):
            values[component] = self.graph.build(
                self.db,
                self.config,
                component.name,
                *component.key,
            )
        return ComponentSnapshot(MappingProxyType(values))

    def rebuild(self, config: Any) -> RebuildReport:
        old_snapshot = self.snapshot
        self.config.set(config)
        self.graph.clear_executions()
        new_snapshot = self._build_snapshot(config)
        self.snapshot = new_snapshot

        old_values = old_snapshot.values
        new_values = new_snapshot.values
        old_targets = set(old_values)
        new_targets = set(new_values)

        added = new_targets - old_targets
        removed = old_targets - new_targets
        common = old_targets & new_targets
        rebuilt = {
            component
            for component in common
            if new_values[component] is not old_values[component]
        }
        reused = common - rebuilt
        executed = {
            ComponentTarget(name, key)
            for name, key in self.graph.executions()
        }
        executed_targets = executed & new_targets

        return RebuildReport(
            added=_sort_targets(added),
            removed=_sort_targets(removed),
            rebuilt=_sort_targets(rebuilt),
            reused=_sort_targets(reused),
            executed=_sort_targets(executed),
            executed_targets=_sort_targets(executed_targets),
        )
