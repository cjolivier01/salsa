# Rebuild Planner Plan

## Goal

Build the first production-facing layer on top of the Python Salsa core: a
planner that can answer which neural-network components are preserved or rebuilt
after a YAML config change.

The immediate target is not a full in-place `nn.Module` morphing system. It is a
deterministic rebuild report with golden tests for representative neural-net
YAML edits.

## Current PR Scope

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

4. Keep the implementation independent of PyTorch runtime classes.
   - The planner works with arbitrary Python component values.
   - PyTorch-specific morphing stays in follow-up work.

## Remaining After This PR

1. Add an in-place PyTorch `MorphableModel` layer.
   - Replace rebuilt submodules in `ModuleDict`/nested modules.
   - Preserve unchanged module identity.
   - Refresh optimizer param groups for replaced parameters.
   - Recompute caches derived from `named_modules()`.

2. Model idempotent post-passes.
   - Adapter injection.
   - Low-precision forward patching.
   - Parameter tying.
   - Annotation naming.
   - State-dict pre-hook registration.
   - Export singleton reset/guard behavior.

3. Integrate the real YAML loader.
   - Recursive merge behavior.
   - Whitelisted list-concat paths.
   - Override hooks.
   - Default-value dependency tracking.
   - Mutation-proof normalized snapshots.

4. Add cross-language conformance tests against Rust Salsa behavior.
   - Shared traces for inputs, dependencies, backdating, cycles, and durability.
   - CI comparison to keep the Python port aligned with Rust semantics.

5. Add deeper Salsa features only if the rebuild layer needs them.
   - Tracked structs.
   - Interning.
   - Accumulators.
   - LRU eviction.
   - Persistence/snapshots.
   - Event tracing.
   - Cancellation and parallel query execution.
