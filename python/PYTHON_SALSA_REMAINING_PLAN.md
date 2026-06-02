# Python Salsa Remaining Plan And Handoff

This document is intentionally self-contained. It should give a future agent
enough context to continue the Python Salsa port without needing the chat
history that produced the current code.

## User Goal

Create a Python-facing Salsa-style incremental build layer for complex
neural-network PyTorch models built from YAML config files.

The practical target is not a literal Rust macro port. The target is the
behavior users need for model-building:

- load and normalize YAML-shaped configs
- track which config fields each model component reads
- rebuild only components whose dependencies changed
- morph an existing `torch.nn.Module` tree in place
- preserve unchanged module identities and optimizer state
- rerun global post-build mutations idempotently
- keep Python query semantics aligned with Rust Salsa where claimed

The Rust crate is still the reference for Salsa semantics, but the Python port
needs dynamic APIs suitable for YAML paths, runtime task names, PyTorch modules,
and custom equality/reuse policies.

## Status

The Python port lives under `python/`. It currently has the first useful
neural-net rebuild loop:

- field-level config dependency tracking
- `ComponentGraph` and `RebuildPlanner`
- golden rebuild-set tests for YAML-shaped neural-net examples
- PyTorch `ModelMorpher` for in-place module replacement/removal
- optimizer repair for morphed parameters
- PyTorch post-pass framework with parameter tying, low-precision hook
  scaffolding, and annotation refresh

Useful files:

- `python/pysalsa/core.py`: database, tracked functions, config input and
  field-level dependency tracking
- `python/pysalsa/rebuild.py`: `ComponentGraph`, `ComponentTarget`,
  `RebuildPlanner`, `RebuildReport`, `ComponentSnapshot`
- `python/pysalsa/pytorch.py`: PyTorch fingerprints, `tracked_module`,
  `ModelMorpher`, optimizer repair
- `python/pysalsa/pytorch_postpass.py`: post-pass framework and current passes
- `python/pysalsa/yaml_config.py`: minimal YAML file merge helper
- `python/tests/test_rebuild_planner.py`: planner golden rebuild-set tests
- `python/tests/test_pytorch_morpher.py`: in-place module morphing tests
- `python/tests/test_pytorch_postpass.py`: post-pass and optimizer regression
  tests
- `python/examples/net_build_incremental.py`: small YAML-style build sketch
- `net_build_pipeline_for_salsa_designers.md`: source design context for the
  real neural-net pipeline

Branch/PR history before this handoff:

1. Python Salsa MVP core was merged into `feature/python-salsa`.
2. Rebuild planner and golden rebuild-set tests were merged.
3. PyTorch `ModelMorpher` and optimizer repair were merged.
4. PyTorch post-pass framework was merged after fixing high/xhigh review
   blockers around removed tied paths, optimizer pruning after parameter tying,
   and alias-aware pass traversal.

The active base branch for future PRs is `feature/python-salsa`, not `master`.

Validation commands used so far:

```text
cd python
python -m pytest -q
PYTHONPATH=. python examples/net_build_incremental.py
```

After the post-pass PR, the full Python suite had `52 passed`.

The remaining work is about turning that into a production-facing path for real
YAML model builds and keeping the Python behavior aligned with Rust Salsa where
that matters.

## Current Behavioral Contracts

### Rebuild Planner

- A `ComponentGraph` declares build functions for named component targets.
- A `RebuildPlanner` owns a config input and component graph.
- `planner.rebuild(next_config)` returns which components were added, removed,
  rebuilt, reused, and executed.
- Equality functions can preserve prior component identity when a freshly built
  value is structurally equivalent.
- Missing config reads with defaults are important: if a key is absent and later
  appears, the dependent component should rebuild.

### PyTorch Morpher

- `ModelMorpher.install(snapshot)` installs the initial modules into a root
  `nn.Module`.
- `ModelMorpher.apply(snapshot, report, optimizer=...)` replaces added/rebuilt
  module paths, removes disappeared paths, and preserves reused identities.
- Component paths must be unique and non-overlapping within a morph.
- Optimizer repair removes stale params and inserts new params, preserving
  parameter-group placement when an optimizer is supplied.
- Distributed/FSDP/DataParallel-wrapped roots are rejected for now.

### PyTorch Post-Passes

- `PostPassManager.apply(...)` runs ordered idempotent passes after a morph.
- `changed_paths` should be added and rebuilt module paths.
- `removed_paths` should be passed separately so post-passes do not recreate
  intentionally removed modules.
- Passing an optimizer lets the manager prune params that post-passes made
  unreachable.
- `force=True` is available when pass state/config changed but the latest morph
  paths do not overlap the pass's module paths.
- `ParameterTyingPass` reports only actual reties in `touched_paths`.
- Low-precision support is currently only a hook/marker scaffold. It wraps and
  restores `forward` idempotently but does not perform real dtype conversion.
- Annotation refresh handles shared-module aliases by visiting duplicate module
  paths and recording all visible annotation paths on shared modules.

## Milestone 1: Remaining Idempotent PyTorch Post-Passes

Goal: make every mutation that currently happens after `MultiTaskNet.__init__`
safe to rerun after a partial rebuild.

### Adapter Injection Pass

- Add an adapter post-pass that wraps supported `Linear` and `Conv2d` modules.
- Use wrapper marker classes so repeated application does not re-wrap adapters
  or recurse into adapter-owned submodules.
- Support include/exclude path filters and regex excludes from YAML-shaped
  `adapter-kwargs`.
- Preserve optimizer behavior when wrappers introduce new trainable parameters.
- Provide a clear policy for trainability changes such as
  `mark_only_adapters_as_trainable`.
- Add tests for:
  - first application wraps expected modules
  - second application is a no-op
  - rebuilt modules are wrapped while existing wrapped modules are not rewrapped
  - excluded adapter internals are not traversed
  - optimizer param groups contain live adapter params only

### State-Dict Hook Pass

- Add an idempotent pre-hook registration pass for tied-stub and export loading
  behavior.
- Track hook handles by stable pass-owned marker attributes.
- Replacing a module should update hooks for that module without duplicating
  hooks on unchanged modules.
- Add tests that count registered hooks before and after repeated application
  and after partial rebuilds.

### Export Singleton Guard Pass

- Model export config state that is currently write-once-or-raise.
- Decide whether partial rebuilds should reuse, replace, or explicitly reset
  singleton state.
- Make the behavior explicit through a pass state object instead of hidden
  process-global mutation.
- Add tests for repeated same-state application, changed-state rejection or
  reset, and interactions with removed export-target modules.

## Milestone 2: Real YAML Loader Integration

Goal: route the real multi-file YAML loading semantics through dependency-aware
inputs instead of plain mutable dicts.

### Loader Behavior

- Implement recursive merge with the same precedence as the existing pipeline:
  CLI arg overrides, dict overrides, later YAML files, earlier YAML files.
- Implement whitelisted list-concat paths.
- Preserve support for override hooks, but record which config paths they read
  and write where possible.
- Normalize loaded configs into mutation-proof snapshots before handing them to
  the planner.
- Add tests for:
  - file precedence
  - dict override precedence
  - list concat whitelist behavior
  - removed keys falling back to defaults
  - override hook effects
  - no caller-visible mutation after snapshot creation

### Default-Value Dependency Tracking

- Ensure `ctx.read(path, default=...)` records a dependency on missing paths.
- If a missing key is later added, rebuild targets that previously consumed the
  default.
- If an explicit key is removed, rebuild targets that should return to the
  default.
- Add golden tests for both add-key and remove-key cases.

## Milestone 3: Build Orchestration API

Goal: provide one high-level API that wires loader, planner, morpher,
post-passes, and optimizer repair together.

- Add an orchestration object, for example `IncrementalModelBuilder`.
- Inputs should include:
  - YAML files or preloaded config dicts
  - component target selector
  - component graph
  - PyTorch path resolver
  - ordered post-pass list
  - optional optimizer
- Outputs should include:
  - rebuilt model reference
  - `RebuildReport`
  - `MorphResult`
  - post-pass results
  - optimizer repair/prune summary
- Add an explicit stable order for post-passes and document which passes are
  topology-changing.
- Add tests that edit one YAML path at a time and assert preserved module
  identity, post-pass reapplication, and optimizer liveness.

## Milestone 4: Cross-Language Salsa Conformance

Goal: keep Python semantics in lock-step with the Rust crate where the Python
port claims Salsa-compatible behavior.

- Add shared trace fixtures for:
  - inputs and revisions
  - tracked dependency recording
  - memo reuse
  - backdating when output equality holds
  - cycle detection
  - durability behavior, if exposed in Python
- Build a small Rust-side trace generator or fixture runner in CI.
- Build a Python-side trace runner that consumes the same fixtures.
- CI should fail when the Python trace diverges from the Rust trace for covered
  semantics.
- Document deliberately unsupported Rust features instead of silently diverging.

## Milestone 5: Deeper Salsa Features, Only As Needed

These should not block the neural-net rebuild path unless a real use case needs
them:

- tracked structs
- interned values
- accumulators
- LRU eviction
- persistence/snapshots
- event tracing
- cancellation
- parallel query execution

Each feature should come with a Rust conformance fixture or a written reason why
the Python behavior intentionally differs.

## Suggested PR Order

1. Adapter injection post-pass and optimizer tests.
2. State-dict hook pass plus export singleton guard behavior.
3. YAML loader integration and mutation-proof snapshots.
4. High-level incremental build orchestration API.
5. Cross-language conformance fixture runner.
6. Optional deeper Salsa features driven by observed rebuild requirements.

## Done Criteria

The port is ready for practical neural-net experiments when:

- changing a head-only YAML value preserves encoders and feature modules
- changing encoder shape rebuilds the dependent feature/head chain only
- adding/removing a task adds/removes only the required modules
- adapter, low-precision, tying, hooks, export state, and annotation passes are
  idempotent after any partial rebuild
- optimizer param groups contain only live model parameters after rebuilds and
  post-passes
- config default changes and missing-key transitions trigger correct rebuilds
- Python query behavior covered by conformance tests matches Rust Salsa traces

## Suggested Next PR: Adapter Injection Post-Pass

This is the highest-value next step because the source design doc calls out
adapter injection as currently non-idempotent and especially risky after partial
rebuilds.

Expected write scope:

- `python/pysalsa/pytorch_postpass.py`
- `python/tests/test_pytorch_postpass.py`
- `python/README.md`
- this plan file, if scope changes

Implementation sketch:

1. Add marker wrapper classes for adapter-wrapped `Linear` and `Conv2d`.
2. Make wrappers expose the same broad forward behavior as the wrapped module
   plus adapter contribution.
3. Add `AdapterState` and `AdapterInjectionPass`.
4. Teach the pass to resolve include/exclude module paths and regex excludes.
5. Stop traversal inside adapter wrapper internals.
6. Return touched paths only for modules that were newly wrapped or whose
   adapter state changed.
7. Use `PostPassManager(...).apply(..., optimizer=optimizer)` to prune stale
   params if wrapping replaces modules.

Minimum tests:

- wraps selected `Linear` and `Conv2d` modules once
- second application is a no-op
- rebuilt selected module is wrapped while existing wrappers are untouched
- excluded regex prevents wrapping adapter internals
- optimizer contains live adapter params and no replaced stale params
- path-scoped manager run applies adapter pass when a touched alias path reaches
  the adapter include path

Keep the first adapter pass intentionally small. It only needs to establish the
idempotent wrapping contract and optimizer safety; it does not need to reproduce
every production LoRA option.
