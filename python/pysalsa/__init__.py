"""Python Salsa-style incremental computation.

The public surface is intentionally small:

* :class:`Database` owns inputs, config inputs, and query memo tables.
* :func:`tracked` memoizes pure functions whose first argument is a database.
* :class:`Input` and :class:`ConfigInput` are mutable roots.

Optional PyTorch helpers live in :mod:`pysalsa.pytorch`.
"""

from .core import (
    MISSING,
    ConfigInput,
    CycleError,
    Database,
    Durability,
    Input,
    TrackedFunction,
    equivalent,
    parse_path,
    stable_key,
    tracked,
)
from .build import ComponentGraph
from .rebuild import ComponentSnapshot, ComponentTarget, RebuildPlanner, RebuildReport, target

__all__ = [
    "MISSING",
    "ConfigInput",
    "ComponentGraph",
    "ComponentSnapshot",
    "ComponentTarget",
    "CycleError",
    "Database",
    "Durability",
    "Input",
    "RebuildPlanner",
    "RebuildReport",
    "TrackedFunction",
    "equivalent",
    "parse_path",
    "stable_key",
    "target",
    "tracked",
]
