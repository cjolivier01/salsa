# Rebuild Planner Plan

## Goal

Build the first production-facing layer on top of the Python Salsa core: a
planner that can answer which neural-network components are preserved or rebuilt
after a YAML config change.

The planner now produces deterministic rebuild reports. The next layer applies
those reports to PyTorch `nn.Module` trees while preserving unchanged object
identity and returning optimizer-repair data.

## Completed So Far

1. Add a `RebuildPlanner` API.
   - Keep a `Database` and `ConfigInput` for a model config.
   - Build a declared set of component targets through `ComponentGraph`.
   - Compare old and new component object identities after a config update.
   - Report added, removed, rebuilt, reused, and executed components.

2. Add target selectors for YAML-shaped model builds.
   - Let callers derive component targets from config snapshots.
   - Support task sets, feature-module DAG pruning, and stable ordering in tests.

3. Add golden rebuild-set tests for neural-net examples.
   - Change one head's `hidden_dim`: rebuild only that head.
   - Change unrelated task metadata: rebuild nothing.
   - Change encoder shape: rebuild encoder, dependent features, and heads.
   - Add a task that requires a previously pruned feature module: add that
     feature module and the new head.
   - Remove a task: remove its head and any now-pruned feature modules.
   - Cover planner API edge cases for supplied `ConfigInput`, dependency
     execution reporting, mixed dynamic keys, and snapshot immutability.

4. Keep the implementation independent of PyTorch runtime classes.
   - The planner works with arbitrary Python component values.

## Current PR Scope

1. Add a PyTorch `ModelMorpher`.
   - Install an initial `ComponentSnapshot` into a root `nn.Module`.
   - Apply a later `RebuildReport` to replace added/rebuilt submodules.
   - Remove targets that disappeared from the planner snapshot.
   - Preserve reused module identities.
   - Refresh caller-provided caches derived from `named_modules()`.
   - Reject distributed/FSDP-wrapped models for now.

2. Return optimizer repair information.
   - Record parameter ids removed by rebuilt/removed modules.
   - Record parameters introduced by rebuilt/added modules.
   - Provide a helper to mutate optimizer param groups and prune stale state.

3. Add PyTorch golden tests.
   - Head-only config change replaces only `model.heads["object_detection"]`.
   - Unrelated metadata change preserves all module identities.
   - Encoder shape change replaces encoder, feature modules, and heads.
   - Added task inserts a new head and newly required feature module.
   - Removed task deletes its head and no-longer-required feature module.
   - Optimizer repair removes old params and appends new params.

## Remaining After This PR

1. Model idempotent post-passes.
   - Adapter injection.
   - Low-precision forward patching.
   - Parameter tying.
   - Annotation naming.
   - State-dict pre-hook registration.
   - Export singleton reset/guard behavior.

2. Integrate the real YAML loader.
   - Recursive merge behavior.
   - Whitelisted list-concat paths.
   - Override hooks.
   - Default-value dependency tracking.
   - Mutation-proof normalized snapshots.

3. Add cross-language conformance tests against Rust Salsa behavior.
   - Shared traces for inputs, dependencies, backdating, cycles, and durability.
   - CI comparison to keep the Python port aligned with Rust semantics.

4. Add deeper Salsa features only if the rebuild layer needs them.
   - Tracked structs.
   - Interning.
   - Accumulators.
   - LRU eviction.
   - Persistence/snapshots.
   - Event tracing.
   - Cancellation and parallel query execution.
