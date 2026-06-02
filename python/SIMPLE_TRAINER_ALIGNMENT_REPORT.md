# Schema-alignment spike report: pysalsa vs simple_trainer / ai

This report accompanies `examples/simple_trainer_schema_spike.py` and
`tests/test_simple_trainer_schema.py`. It records what aligned, what broke,
and what worked-with-friction when driving the real
`../simple_trainer/configs/lenet.yaml` through pysalsa's current primitives.

The spike is intentionally narrow: it covers the **composed-model path** of
`simple_trainer.models.hydranet.HydraNet.from_config` (the same path the
production `../ai/training/tasks/hydra/hydra.py` `HydraNet.from_config`
expands to for the multi-task case). It does not exercise the
single-class `train_args.model: <dotted.path>` path, nor any of the
post-pass surface (adapter / low-precision / export-singleton).

## TL;DR

Pysalsa's *primitives* (Database, ConfigInput, ComponentGraph,
RebuildPlanner, ModelMorpher) work against the real schema with no code
changes. The pain is concentrated in the *examples and tests*, which use a
synthetic vocabulary (`run_args` / `encoders` / `feature_modules` /
`tasks`) that does not appear in either real target. The next milestones
will accumulate friction if more post-passes ship against the synthetic
schema before the example layer is realigned.

## What aligned

1. **Database / ConfigInput / RebuildPlanner**: all worked unchanged against
   the real `train_args` / `shared_trunks` / `pickles` vocabulary. Path
   reads like `ctx.read(("pickles", "lenet"))` and
   `ctx.read(("shared_trunks", trunk_name))` recorded field-level
   dependencies correctly, and the rebuild plan was exact (head rebuilt,
   trunk reused) when `pickles.lenet.hdim` was changed.

2. **ComponentGraph builder shape**: the `(ctx, *key)` signature accepted
   the dynamic `(trunk_name,)` / `(task_name,)` keys naturally. No new
   abstractions were needed.

3. **`load_yaml_files`**: parsed `simple_trainer/configs/lenet.yaml`
   verbatim without YAML-tag or merge-key issues. It also handles the
   PyYAML 6.0+ env.

4. **ModelMorpher install + apply**: worked through `nn.Sequential` parents
   even though the morpher was designed against `nn.ModuleDict`-shaped
   parents. `setattr(empty_seq, "0", trunk)` populates `_modules["0"]`
   correctly, so an empty Sequential is a valid morphable container.

5. **Optimizer repair**: produced the expected swap (head params replaced,
   trunk params preserved) and `refresh_optimizer` left the optimizer with
   exactly the live parameters.

## What broke or needed workarounds

### §1. Schema vocabulary mismatch (high impact, examples-only fix)

The synthetic schema baked into `examples/net_build_incremental.py` and
`tests/test_config_build_pipeline.py` / `tests/test_pytorch_morpher.py`
uses keys that **do not exist** in either real target:

| Synthetic example         | simple_trainer        | ai (`hydra.py`)             |
|---------------------------|-----------------------|-----------------------------|
| `run_args`                | `train_args`          | `train_args`                |
| `encoders` / `encoder_params` | `backbone` (or none for LeNet) | `backbone` keyed by `backbone-key` |
| `feature_modules`         | `shared_trunks`       | `shared_trunks`             |
| `tasks`                   | `pickles`             | `pickles`                   |
| `tie_encoder_params`      | (none)                | `tie-encoder-params`        |
| `export_config.extra_export_params` | (none)      | `txp.extra_export_params`   |
| `mixed_precision`         | (none)                | `low_precision`             |

Impact: the **plan's Milestone 1 post-passes** (adapter, state-dict hook,
export singleton) are about to ship against the synthetic schema. They
will work but will need translation when ai integration begins. Cheaper to
realign the examples now.

Suggested fix: rename the example/test config keys to the real vocabulary
and update the synthetic test fixtures accordingly. This is roughly a
search-and-replace in `examples/net_build_incremental.py`,
`tests/test_config_build_pipeline.py`, `tests/test_pytorch_morpher.py`,
and `tests/test_pytorch_postpass.py`. The pysalsa primitives themselves
are schema-neutral and need no changes.

### §2. HydraNet's `inner = Sequential(trunk, head)` produces positional paths

simple_trainer's `_build_composed_model` returns
`nn.Sequential(trunk, head)` and stores it as `self.inner` on HydraNet.
The morpher therefore sees paths `("inner", "0")` for the trunk and
`("inner", "1")` for the head — purely positional names tied to construction
order.

Compared to the synthetic example's `encoders["image"]`,
`feature_modules["bev"]`, `heads["object_detection"]` — *named slots* keyed
by the config — Sequential's positional slots have a sharp edge: any
component that inserts itself in the middle (e.g. a per-task neck between
trunk and head) shifts downstream paths. The morpher would treat that as
*remove `inner.1`, add `inner.2`* rather than *insert at `inner.1`,
shift head to `inner.2`*.

Suggested fix: when the spike is promoted to an integration, wrap
HydraNet's `inner` in a `ModuleDict` keyed by component role (e.g.
`{"trunk": trunk, "head": head}`) rather than a positional Sequential.
This requires changing simple_trainer's HydraNet (or living with the
positional fragility). The cleanest long-term fix is to make ModelMorpher
support a "named-slot Sequential" wrapper class, but that is significant
new surface — flag for design discussion.

### §3. No registry / no class lookup primitive

simple_trainer uses module-level dict registries (`_TRUNKS`, `_TASKS`)
populated by `@register_trunk` / `@register_task` decorators at import
time. The ai repo uses the same pattern at much larger scale (the design
doc estimates "hundreds of entries" in the task registry alone).

Pysalsa has no equivalent. The spike works around this by hard-coding two
inline dicts (`_TRUNK_REGISTRY`, `_TASK_HEAD_REGISTRY`). For production
use the orchestration API (Milestone 3) will need to either:
- accept user-supplied registries as a constructor argument, or
- treat registry lookup as a tracked query (so registry edits invalidate),
  or
- declare registries out of scope and require the user to wrap their own.

### §4. `shared_trunks` value is a *list* of specs

In `lenet.yaml`:

```yaml
shared_trunks:
  lenet_features:
    - class: LeNetFeatures
      params: {}
```

simple_trainer's loader composes multi-spec lists into
`nn.Sequential(*modules)`. The spike asserts single-spec only
(`NotImplementedError` otherwise). A production-faithful pass would have
to:
- either expose each spec as an indexed sub-component (e.g. component key
  `("trunk", "lenet_features", 0)`),
- or always memoize at the trunk-name level and let the builder iterate
  internally (losing per-spec dependency tracking).

This is the same shape problem the synthetic example has with
`feature_modules` values being `dict | list[dict]` — solved there by
reading index 0. The same pattern works here, but the framework should
codify the convention rather than letting each integration re-discover it.

### §5. Task-instance-bound head construction

simple_trainer's `Task.build_head()` is an instance method on the Task
class — it reads `self.params` (set by `Task.__init__(params)`). The
production ai equivalent (`TaskSpec.head()` per the design doc §6.5) is
the same shape. Pysalsa cannot directly memoize through a Task instance
because `stable_key` falls back to `id(value)` for arbitrary mutable
objects — the memo key would change if the Task is reconstructed even
when its params are identical.

The spike bypasses Task entirely (`_TASK_HEAD_REGISTRY` reads
`pickles.{task_name}` and returns a head directly). For production this
needs either:
- a Task wrapper that's hashable by `(class, frozen-params)`, or
- a documented convention that head builders read raw config dicts and
  task objects are constructed downstream of the rebuild.

The design doc's §12 open complication #2 ("`task.head()` returns a
fresh module each call") is the same problem viewed from the other side.

### §6. Single-task / single-trunk constraint is not expressible in the graph

`lenet.yaml` has one task (`lenet`) requiring exactly one trunk
(`lenet_features`). simple_trainer's `_build_composed_model` raises
`NotImplementedError` for multi-trunk composition. The spike's
`_select_targets` enforces the same constraint imperatively.

This is fine, but exposes a gap: pysalsa has no notion of cross-component
constraints. A multi-task config that selected two heads would silently
produce two head components both targeting `inner.1` — the morpher would
crash at `_validate_non_overlapping_paths` rather than producing a clear
"this schema does not support multi-task" error. The orchestration API
should catch this at planner construction.

### §7. Root `forward()` and dict-shaped output are out of band

`nn.Module` requires the root to implement `forward()`. The morpher
operates on the *module tree*; it has no view of `forward()`. A morph
that swaps in a head with mismatched input shape (e.g. a trunk that
outputs 256 channels and a head expecting 128) will pass
`_validate_non_overlapping_paths` and `_set_module`, then crash at the
next forward pass with a shape mismatch.

The synthetic example dodges this by using fake `Block` modules whose
`forward` is `pass`. Real HydraNet returns `{"yv_hat": self.inner(x)}` —
a dict output the trainer downstream relies on. The orchestration API
should at minimum offer a "smoke forward" hook that runs a dummy input
through the morphed model and surfaces shape errors at morph time.

## Implications for the existing plan (PYTHON_SALSA_REMAINING_PLAN.md)

The plan's suggested next PR is the Adapter Injection Post-Pass against
the synthetic schema. After this spike:

- **The adapter pass itself doesn't change** — it operates on
  `nn.Linear` / `nn.Conv2d` regardless of where they live in the tree.
  Schema vocabulary doesn't affect post-pass code.

- **The adapter pass *tests* will solidify the synthetic schema.** Every
  new post-pass test that uses `feature_modules` / `tasks` / `encoders` as
  fixture keys widens the cleanup surface when ai integration begins.

- **Recommendation**: before adapter injection, rename the synthetic-schema
  keys in `examples/net_build_incremental.py`,
  `tests/test_config_build_pipeline.py`, `tests/test_pytorch_morpher.py`,
  and `tests/test_pytorch_postpass.py` to the production vocabulary.
  This is a mechanical rename that touches no pysalsa source.

- **Milestone 3 (orchestration API)** should subsume the registry /
  task-binding / shape-smoke-test concerns surfaced in §3 / §5 / §7. The
  API contract should be clear about what it owns vs what the user
  provides.

## Open questions for the design

These need a decision before Milestone 1 is closed:

1. Does ModelMorpher gain first-class support for **named-slot
   containers** (e.g. a `NamedSequential` ModuleDict wrapper that
   preserves Sequential's forward semantics), or do we require users to
   restructure HydraNet's `inner` into a ModuleDict?

2. Should pysalsa **own a registry primitive** or stay out of registry
   business and require integrators to wrap their own?

3. Is **Task-instance-bound head construction** a contract we accept
   (with a documented hashing convention), or do we force head builders
   to be free functions that read raw config?

4. Where does **smoke-forward shape validation** live? Inside
   ModelMorpher.apply? In a new orchestration object? As an opt-in
   `PostPass` named `forward_smoke`?

Answering these informs whether Milestone 1's post-passes get written
against the synthetic schema or wait for a small schema-realignment PR
first.
