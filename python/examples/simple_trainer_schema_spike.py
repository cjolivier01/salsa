"""Schema-alignment spike: drive simple_trainer's HydraNet through pysalsa.

The synthetic schema used elsewhere in pysalsa (``run_args`` / ``encoders`` /
``feature_modules`` / ``tasks``) does not match either of the real targets:

* ``../simple_trainer`` (the small, runnable sample) uses ``train_args`` /
  ``shared_trunks`` / ``pickles`` / ``required_trunks``.
* ``../ai`` (the production target) uses the same vocabulary in
  ``training/tasks/hydra/hydra.py`` plus ``backbone`` / ``txp`` /
  ``low_precision``.

This spike loads ``../simple_trainer/configs/lenet.yaml`` *verbatim*, mirrors
``HydraNet.from_config``'s composed-model construction
(``nn.Sequential(trunk, head)`` under ``self.inner``), and drives the build
through pysalsa's ``Database`` + ``ComponentGraph`` + ``RebuildPlanner`` +
``ModelMorpher``. The goal is not to ship simple_trainer integration — it is
to surface every assumption in pysalsa that needs to flex for the real
schema.

Findings are documented in ``SIMPLE_TRAINER_ALIGNMENT_REPORT.md`` next to
this file. The header comments in each section below cross-reference the
matching finding.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from pysalsa import ComponentGraph, RebuildPlanner, target
from pysalsa.pytorch import ModelMorpher, modules_equivalent
from pysalsa.yaml_config import load_yaml_files


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMPLE_TRAINER_LENET_YAML = REPO_ROOT.parent / "simple_trainer" / "configs" / "lenet.yaml"


# ---------------------------------------------------------------------------
# Module shapes mirroring simple_trainer/models/letnet.py and tasks.py.
# Re-defined inline so this spike has no runtime dependency on simple_trainer.
# The classes are byte-equivalent to the simple_trainer originals — see
# ../simple_trainer/models/letnet.py and ../simple_trainer/models/tasks.py.
# ---------------------------------------------------------------------------


class LeNetFeatures(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class LeNetClassifier(nn.Sequential):
    def __init__(self, *, hdim: int = 84, num_classes: int = 10) -> None:
        super().__init__(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, hdim),
            nn.ReLU(inplace=True),
            nn.Linear(hdim, num_classes),
        )


# ---------------------------------------------------------------------------
# Finding §3: simple_trainer pushes class instantiation through a registry.
# pysalsa has no equivalent. The spike provides a local registry to keep the
# example faithful to the production shape without dragging in
# simple_trainer's models.registry module.
# ---------------------------------------------------------------------------

_TRUNK_REGISTRY: dict[str, type[nn.Module]] = {"LeNetFeatures": LeNetFeatures}
_TASK_HEAD_REGISTRY: dict[str, callable] = {
    "lenet": lambda params: LeNetClassifier(
        hdim=int(params.get("hdim", 84)),
        num_classes=int(params.get("num_classes", 10)),
    )
}


# ---------------------------------------------------------------------------
# Finding §1: the synthetic ``run_args``/``encoders``/``feature_modules``/
# ``tasks`` vocabulary does not appear anywhere in simple_trainer or ai. The
# tracked builders below speak the production vocabulary verbatim.
# ---------------------------------------------------------------------------

# ``ctx.read(...)`` paths reference simple_trainer's lenet.yaml keys:
#     train_args.{...}
#     shared_trunks.{trunk_name}.0.{class|params}
#     pickles.{task_name}.{hdim|num_classes|required_trunks}


def _component_graph() -> ComponentGraph:
    """Mirror simple_trainer.models.hydranet._build_composed_model."""

    graph = ComponentGraph()

    @graph.component("trunk", equals=modules_equivalent)
    def trunk(ctx, trunk_name: str) -> nn.Module:
        # Finding §4: simple_trainer's ``shared_trunks`` value is a *list of
        # specs*; only the single-spec case is exercised by lenet.yaml. The
        # multi-spec sequential case (``nn.Sequential(*modules)``) exists in
        # simple_trainer's loader but is not exercised here. A
        # production-faithful pass would have to read the list length first.
        specs = ctx.read(("shared_trunks", trunk_name))
        if not isinstance(specs, list) or len(specs) != 1:
            raise NotImplementedError(
                f"spike only covers single-spec trunks; got {specs!r}"
            )
        spec = specs[0]
        cls = _TRUNK_REGISTRY[spec["class"]]
        return cls(**spec.get("params", {}))

    @graph.component("head", equals=modules_equivalent)
    def head(ctx, task_name: str) -> nn.Module:
        # Finding §5: simple_trainer's head construction is bound to the
        # *Task* object (Task.build_head() in models/tasks.py). pysalsa cannot
        # memoize through a Task instance without a wrapper — this spike
        # bypasses Task entirely and reads ``pickles.{task_name}`` directly.
        params = ctx.read(("pickles", task_name))
        head_factory = _TASK_HEAD_REGISTRY[task_name]
        return head_factory(params)

    return graph


def _select_targets(config: dict) -> list:
    """Mirror simple_trainer's single-task / single-trunk constraint."""

    pickles = config.get("pickles") or {}
    if not pickles:
        raise ValueError("yaml config must define at least one task under `pickles`")
    # Finding §6: simple_trainer's _build_composed_model enforces exactly one
    # task with exactly one ``required_trunks`` entry. The synthetic schema's
    # multi-task / DAG feature_modules has no analogue here.
    task_names = list(pickles)
    if len(task_names) != 1:
        raise NotImplementedError("spike only covers single-task pickles")
    (task_name,) = task_names
    required = pickles[task_name].get("required_trunks") or []
    if len(required) != 1:
        raise NotImplementedError("spike only covers exactly-one-trunk tasks")
    (trunk_name,) = required
    return [target("trunk", trunk_name), target("head", task_name)]


# ---------------------------------------------------------------------------
# Finding §2: HydraNet wraps its built modules in ``self.inner =
# nn.Sequential(trunk, head)``. There are no stable named slots like
# ``encoders["image"]`` or ``heads["object_detection"]`` — the morpher's
# implicit assumption (ModuleDict-like parents) does not hold.
#
# ModelMorpher *does* still work because Sequential exposes children via
# ``getattr/setattr`` on numeric-string names ("0", "1"), but the resulting
# paths are positional and depend on construction order. Adding a head in
# the middle of a Sequential would shift positional paths under nodes that
# weren't supposed to change.
# ---------------------------------------------------------------------------


class HydraNetLike(nn.Module):
    """A faithful mirror of simple_trainer's HydraNet wrapper.

    Mirrors:
    * the ``self.inner`` indirection
    * the dict-shaped forward output (simple_trainer returns
      ``{"yv_hat": self.inner(x)}``)

    Omits checkpoint round-tripping (the ``inner.``-prefix state_dict hook),
    which is out of scope for the structural spike.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``inner`` must exist as a Module before the morpher can ``setattr``
        # children onto it. Empty Sequential satisfies that — children are
        # added in order during ``install``.
        self.inner = nn.Sequential()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Finding §7: ``nn.Module.forward`` is required for the *root* to be
        # callable, but the morpher knows nothing about forward(). A morph
        # that swaps in a head with mismatched input shape would type-check
        # at install time and crash only on the next forward pass.
        return {"yv_hat": self.inner(x)}


def _path_for_target(component) -> tuple[str, ...]:
    # ``inner.0`` = trunk slot, ``inner.1`` = head slot.
    # Finding §2: paths are *positional* under Sequential. Adding a third
    # component in the middle (e.g. a per-task neck) would shift the head
    # to ``inner.2``, which the morpher would treat as "removed inner.1 +
    # added inner.2" rather than "head moved".
    if component.name == "trunk":
        return ("inner", "0")
    if component.name == "head":
        return ("inner", "1")
    raise KeyError(component)


def _build_and_morph(yaml_path: Path) -> tuple[HydraNetLike, torch.optim.Optimizer]:
    config = load_yaml_files(yaml_path)
    graph = _component_graph()
    planner = RebuildPlanner(graph, config, _select_targets)

    model = HydraNetLike()
    morpher = ModelMorpher(model, _path_for_target)
    morpher.install(planner.snapshot)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Smoke forward pass through the composed model.
    model(torch.zeros(1, 3, 32, 32))

    # Edit one production-schema key (``pickles.lenet.hdim``) and confirm
    # only the head rebuilds while the trunk identity is preserved.
    new_config = planner.config.snapshot()
    new_config["pickles"]["lenet"]["hdim"] = 96
    old_trunk = model.inner[0]
    old_head = model.inner[1]
    report = planner.rebuild(new_config)
    morph_result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
    morph_result.refresh_optimizer(optimizer)

    assert {component.name for component in report.rebuilt} == {"head"}, report.rebuilt
    assert {component.name for component in report.reused} == {"trunk"}, report.reused
    assert model.inner[0] is old_trunk, "trunk should retain identity"
    assert model.inner[1] is not old_head, "head should be a new module"

    return model, optimizer


def main() -> None:
    if not SIMPLE_TRAINER_LENET_YAML.exists():
        raise SystemExit(
            f"missing simple_trainer config at {SIMPLE_TRAINER_LENET_YAML} — this "
            "spike requires the sibling simple_trainer checkout"
        )
    model, _ = _build_and_morph(SIMPLE_TRAINER_LENET_YAML)
    print("schema-alignment spike succeeded")
    print(f"  model tree: {[name for name, _ in model.named_children()]}")
    print(f"  inner tree: {[name for name, _ in model.inner.named_children()]}")


if __name__ == "__main__":
    main()
