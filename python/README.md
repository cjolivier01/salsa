# pysalsa-port

This is a Python-first port of Salsa's core incremental-computation model.
It is intentionally not a direct translation of the Rust macro system.
The useful behavior for Python model building is:

- mutable input roots held in a database
- pure tracked functions that memoize by call key
- dependency tracking while tracked functions run
- lazy red-green-style verification after input changes
- backdating when a recomputed value is equivalent to the old value
- field-level config dependencies for large YAML-derived dictionaries
- optional PyTorch helpers for preserving module identity when structure is unchanged

The package lives under `python/` so it can evolve without disturbing the Rust crate.

## Minimal query example

```python
from pysalsa import Database, tracked

db = Database()
source = db.input(22)
events = []

@tracked
def half(db, value):
    events.append("half")
    return value.get() // 2

@tracked
def doubled_half(db, value):
    events.append("doubled_half")
    return half(db, value) * 2

assert doubled_half(db, source) == 22
events.clear()

source.set(23)
assert doubled_half(db, source) == 22
assert events == ["half"]
```

Changing `22` to `23` re-runs `half`, but the value is still `11`.
The old value is backdated, so `doubled_half` is not re-run.

## Config-driven model building

Use `Database.config(...)` for YAML-derived dictionaries.
Tracked builders should read the narrowest paths they actually consume:

```python
from pysalsa import Database, tracked
from pysalsa.pytorch import modules_equivalent

db = Database()
config = db.config(loaded_yaml_dict)

@tracked(equals=modules_equivalent)
def build_head(db, cfg, task_name):
    head_config = cfg.read(("tasks", task_name, "head_config"))
    input_mapping = cfg.read(("tasks", task_name, "input_mapping"), default={})
    shapes = {
        alias: feature_shape(db, cfg, source_path)
        for source_path, alias in input_mapping.items()
    }
    return make_torch_head(task_name, head_config, shapes)
```

If `tasks.object_detection.head_config.hidden_dim` changes, only queries that read
that path or a parent slice need to be revalidated. If `build_head` returns a
new module with the same structural fingerprint, the cached old module is reused
so existing weights and optimizer references can survive where appropriate.

## Rebuild planning

`RebuildPlanner` builds named `ComponentGraph` targets for a config snapshot and
then reports which targets were preserved or rebuilt after a later config
snapshot:

```python
from pysalsa import ComponentGraph, RebuildPlanner, target

graph = ComponentGraph()

@graph.component("head")
def build_head(ctx, task_name):
    return make_head(ctx.read(("tasks", task_name, "head_config")))

def targets(config):
    for task_name in sorted(config["tasks"]):
        yield target("head", task_name)

planner = RebuildPlanner(graph, loaded_yaml_dict, targets)
report = planner.rebuild(next_yaml_dict)
print(report.to_golden()["rebuild_set"])
```

`report.executed` includes every component builder that actually ran, including
dependencies outside the selected target set. `report.executed_targets` is the
selected-target-only subset.

The first planner tests use synthetic neural-net YAML examples to lock down
expected rebuild sets for head-only edits, unrelated metadata edits, encoder
shape cascades, and adding tasks that require previously pruned feature modules.

## PyTorch morphing

`ModelMorpher` applies planner snapshots to an existing `nn.Module` tree:

```python
from pysalsa.pytorch import ModelMorpher

def path_for(component):
    if component.name == "head":
        return ("heads", component.key[0])
    raise KeyError(component)

morpher = ModelMorpher(model, path_for, refresh=lambda root: root.rebuild_caches())
morpher.install(planner.snapshot)

report = planner.rebuild(next_yaml_dict)
result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
result.refresh_optimizer(optimizer)
```

The morpher replaces only added and rebuilt modules, removes disappeared
targets, leaves reused modules untouched, and returns parameter ids/objects
needed to repair optimizer param groups.
Pass the optimizer to `apply` when preserving per-group settings matters.
Component paths must be unique and non-overlapping; nested module paths are
supported, but one component path cannot be the parent of another component path
in the same morph.

## PyTorch post-passes

`PostPassManager` runs idempotent tree mutations after a morph:

```python
from pysalsa.pytorch_postpass import (
    AnnotationPass,
    LowPrecisionPatchPass,
    LowPrecisionState,
    ParameterTyingPass,
    PostPassManager,
)

manager = PostPassManager(
    [
        ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]]),
        LowPrecisionPatchPass(LowPrecisionState(tag="qat"), module_paths=["heads"]),
        AnnotationPass("MultiTaskNet"),
    ]
)

changed_paths = [*result.added_paths, *result.rebuilt_paths]
manager.apply(
    model,
    changed_paths=changed_paths,
    removed_paths=result.removed_paths,
    optimizer=optimizer,
)
```

The included passes are safe to rerun. Parameter tying restores aliases after a
partial rebuild without recreating removed aliases. When an optimizer is passed
to the manager, it prunes parameters that post-passes made unreachable, such as
discarded alias parameters after tying.

Low-precision patching is currently a stable hook/marker scaffold: it wraps
`forward` once, stores caller state, and restores the original method when
disabled, but it does not perform dtype conversion by itself yet. Annotation
refresh visits duplicate module aliases and records all visible annotation paths
on shared modules.

Use `force=True` when a post-pass's own configuration or state changed and the
latest morph paths do not overlap the pass's module paths.

## Why not bind the Rust crate directly?

The Rust implementation gets much of its ergonomics from macros and Rust's type
system: `#[salsa::input]`, `#[salsa::tracked]`, tracked structs, interned ids,
and generated storage. A Python extension could eventually reuse lower-level
Rust storage, but Python still needs a dynamic query layer for YAML paths,
runtime task names, PyTorch modules, and custom equality/reuse policies.
This package starts with that Python layer.
