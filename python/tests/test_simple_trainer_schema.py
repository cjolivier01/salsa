"""Schema-alignment tests for simple_trainer's HydraNet shape.

These tests pin the schema-spike's behavior so that future PRs that change
pysalsa primitives must keep working against ``../simple_trainer``'s real
schema (``train_args`` / ``shared_trunks`` / ``pickles``), not just the
synthetic vocabulary in ``test_config_build_pipeline.py`` and
``test_pytorch_morpher.py``.

The tests deliberately re-define LeNet's layers inline instead of importing
``simple_trainer``. Sibling-repo imports would couple CI to a separate
checkout; the structural alignment we care about is the *config schema* and
the *HydraNet module-tree shape*, both of which we reproduce faithfully here.

See ``SIMPLE_TRAINER_ALIGNMENT_REPORT.md`` for the findings these tests pin.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from pysalsa import ComponentGraph, RebuildPlanner, target
from pysalsa.pytorch import ModelMorpher, modules_equivalent


# Mirror of simple_trainer.models.letnet.LeNetFeatures / LeNetClassifier.
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


# Mirror of simple_trainer.models.hydranet.HydraNet (composed-model path only,
# state_dict prefix hooks omitted).
class HydraNetLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"yv_hat": self.inner(x)}


_TRUNK_REGISTRY: dict[str, type[nn.Module]] = {"LeNetFeatures": LeNetFeatures}


def _head_for_task(task_name: str, params: dict[str, Any]) -> nn.Module:
    if task_name == "lenet":
        return LeNetClassifier(
            hdim=int(params.get("hdim", 84)),
            num_classes=int(params.get("num_classes", 10)),
        )
    raise KeyError(task_name)


def _lenet_config() -> dict[str, Any]:
    """Verbatim copy of ../simple_trainer/configs/lenet.yaml."""

    return {
        "train_args": {
            "base-learning-rate": 1.0e-3,
            "optimizer": "adam",
            "batch-size-per-gpu": 64,
            "max-iters": 1000,
            "save-path": "./lenet_cifar10.pt",
            "data-workers": 2,
        },
        "shared_trunks": {
            "lenet_features": [{"class": "LeNetFeatures", "params": {}}],
        },
        "pickles": {
            "lenet": {
                "hdim": 84,
                "num_classes": 10,
                "required_trunks": ["lenet_features"],
            },
        },
    }


def _component_graph() -> ComponentGraph:
    graph = ComponentGraph()

    @graph.component("trunk", equals=modules_equivalent)
    def trunk(ctx, trunk_name: str) -> nn.Module:
        specs = ctx.read(("shared_trunks", trunk_name))
        spec = specs[0]
        cls = _TRUNK_REGISTRY[spec["class"]]
        return cls(**spec.get("params", {}))

    @graph.component("head", equals=modules_equivalent)
    def head(ctx, task_name: str) -> nn.Module:
        params = ctx.read(("pickles", task_name))
        return _head_for_task(task_name, params)

    return graph


def _select_targets(config: dict) -> list:
    pickles = config["pickles"]
    (task_name,) = list(pickles)
    (trunk_name,) = pickles[task_name]["required_trunks"]
    return [target("trunk", trunk_name), target("head", task_name)]


def _path_for_target(component) -> tuple[str, ...]:
    if component.name == "trunk":
        return ("inner", "0")
    if component.name == "head":
        return ("inner", "1")
    raise KeyError(component)


def _build_initial() -> tuple[HydraNetLike, ModelMorpher, RebuildPlanner, torch.optim.Optimizer]:
    config = _lenet_config()
    graph = _component_graph()
    planner = RebuildPlanner(graph, config, _select_targets)

    model = HydraNetLike()
    morpher = ModelMorpher(model, _path_for_target)
    morpher.install(planner.snapshot)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train_args"]["base-learning-rate"])
    return model, morpher, planner, optimizer


def test_initial_build_populates_inner_sequential_with_real_schema() -> None:
    model, _, planner, _ = _build_initial()

    assert list(model.inner) == [
        planner.snapshot.values[target("trunk", "lenet_features")],
        planner.snapshot.values[target("head", "lenet")],
    ]
    assert isinstance(model.inner[0], LeNetFeatures)
    assert isinstance(model.inner[1], LeNetClassifier)


def test_initial_build_supports_smoke_forward_with_dict_output() -> None:
    model, *_ = _build_initial()

    output = model(torch.zeros(2, 3, 32, 32))

    assert set(output) == {"yv_hat"}
    assert output["yv_hat"].shape == (2, 10)


def test_pickles_hdim_change_rebuilds_only_head_and_preserves_trunk() -> None:
    model, morpher, planner, optimizer = _build_initial()
    old_trunk = model.inner[0]
    old_head = model.inner[1]
    old_trunk_params = {id(p) for p in old_trunk.parameters()}
    old_head_params = {id(p) for p in old_head.parameters()}

    next_config = planner.config.snapshot()
    next_config["pickles"]["lenet"]["hdim"] = 96

    report = planner.rebuild(next_config)
    morph_result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
    morph_result.refresh_optimizer(optimizer)

    assert [c.name for c in report.rebuilt] == ["head"]
    assert [c.name for c in report.reused] == ["trunk"]
    assert model.inner[0] is old_trunk
    assert model.inner[1] is not old_head
    assert isinstance(model.inner[1], LeNetClassifier)

    live_params = {id(p) for p in model.parameters()}
    optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimizer_params == live_params
    # Trunk params preserved by identity, head params fully swapped.
    assert old_trunk_params <= optimizer_params
    assert old_head_params.isdisjoint(optimizer_params)


def test_unrelated_train_args_edit_does_not_rebuild_modules() -> None:
    model, morpher, planner, optimizer = _build_initial()
    old_trunk = model.inner[0]
    old_head = model.inner[1]

    next_config = planner.config.snapshot()
    next_config["train_args"]["max-iters"] = 9999

    report = planner.rebuild(next_config)
    morph_result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
    morph_result.refresh_optimizer(optimizer)

    assert report.rebuilt == ()
    assert report.added == ()
    assert report.removed == ()
    assert model.inner[0] is old_trunk
    assert model.inner[1] is old_head


def test_trunk_class_change_rebuilds_trunk_and_cascades_via_modules_equivalent() -> None:
    """Swapping the trunk *class* triggers a trunk rebuild.

    The head doesn't rebuild here because its config slice
    (``pickles.lenet``) is unchanged and the head builder doesn't depend on
    the trunk's output shape — simple_trainer's schema decouples them at
    YAML level. This matches simple_trainer's actual behavior: the head's
    input dim is hard-coded at ``16 * 5 * 5``.

    Finding §7 in SIMPLE_TRAINER_ALIGNMENT_REPORT.md: this is exactly the
    case where a smoke-forward shape check would catch a real mismatch
    that pysalsa cannot infer from config slices alone.
    """

    class FatLeNetFeatures(nn.Sequential):
        def __init__(self) -> None:
            super().__init__(
                nn.Conv2d(3, 12, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(12, 16, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

    _TRUNK_REGISTRY["FatLeNetFeatures"] = FatLeNetFeatures
    try:
        model, morpher, planner, optimizer = _build_initial()
        old_head = model.inner[1]

        next_config = planner.config.snapshot()
        next_config["shared_trunks"]["lenet_features"][0]["class"] = "FatLeNetFeatures"

        report = planner.rebuild(next_config)
        morph_result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
        morph_result.refresh_optimizer(optimizer)

        assert [c.name for c in report.rebuilt] == ["trunk"]
        assert [c.name for c in report.reused] == ["head"]
        assert isinstance(model.inner[0], FatLeNetFeatures)
        assert model.inner[1] is old_head
    finally:
        del _TRUNK_REGISTRY["FatLeNetFeatures"]
