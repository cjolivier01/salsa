from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def recursive_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update dictionaries; other values replace."""

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_files(paths: str | Path | Iterable[str | Path]) -> dict[str, Any]:
    """Load and recursively merge one or more YAML files.

    Later files win. PyYAML is imported lazily so the core package has no YAML
    dependency.
    """

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("pysalsa.yaml_config requires PyYAML to be installed") from exc

    if isinstance(paths, (str, Path)):
        iterable = [paths]
    else:
        iterable = list(paths)

    merged: dict[str, Any] = {}
    for path in iterable:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"YAML root must be a mapping: {path}")
        recursive_update(merged, loaded)
    return merged
