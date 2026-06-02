from __future__ import annotations

from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")

from pysalsa import ComponentSnapshot, RebuildReport, target
from pysalsa.pytorch import ModelMorpher, _set_module
from pysalsa.pytorch_postpass import (
    AnnotationPass,
    LowPrecisionPatchPass,
    LowPrecisionState,
    ParameterTyingPass,
    PostPassManager,
)


def same_bound_method(left, right) -> bool:
    return getattr(left, "__self__", None) is getattr(right, "__self__", None) and getattr(
        left,
        "__func__",
        None,
    ) is getattr(right, "__func__", None)


class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders = torch.nn.ModuleDict(
            {
                "cam_a": torch.nn.Linear(2, 2),
                "cam_b": torch.nn.Linear(2, 2),
            }
        )
        self.heads = torch.nn.ModuleDict({"task": torch.nn.Linear(2, 1)})


def optimizer_param_ids(optimizer) -> set[int]:
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def test_parameter_tying_is_idempotent_and_restores_alias_after_replacement() -> None:
    model = Net()
    pass_ = ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]])

    result = pass_.apply(model)
    assert result.labels() == ("encoders.cam_b",)
    assert model.encoders["cam_b"] is model.encoders["cam_a"]

    old_primary = model.encoders["cam_a"]
    assert pass_.apply(model).labels() == ()
    assert model.encoders["cam_b"] is old_primary

    _set_module(model, ("encoders", "cam_b"), torch.nn.Linear(2, 2))
    assert model.encoders["cam_b"] is not old_primary

    pass_.apply(model)
    assert model.encoders["cam_b"] is old_primary


def test_parameter_tying_does_not_recreate_removed_alias_or_raise_for_removed_primary() -> None:
    pass_ = ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]])
    model = Net()
    pass_.apply(model)

    del model.encoders["cam_b"]
    result = pass_.apply(model, changed_paths=["encoders.cam_b"], removed_paths=["encoders.cam_b"])

    assert result.labels() == ()
    assert "cam_b" not in model.encoders

    model = Net()
    pass_.apply(model)
    del model.encoders["cam_a"]

    result = pass_.apply(model, changed_paths=["encoders.cam_a"], removed_paths=["encoders.cam_a"])

    assert result.labels() == ()
    assert "cam_a" not in model.encoders
    assert "cam_b" in model.encoders


def test_post_pass_manager_prunes_optimizer_after_tying_rebuilt_alias() -> None:
    model = Net()
    tying = ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]])
    tying.apply(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    primary_param_ids = {id(param) for param in model.encoders["cam_a"].parameters()}

    def path_for(component):
        return ("encoders", component.key[0])

    rebuilt = target("encoder", "cam_b")
    new_alias = torch.nn.Linear(2, 2)
    snapshot = ComponentSnapshot(MappingProxyType({rebuilt: new_alias}))
    report = RebuildReport(
        added=(),
        rebuilt=(rebuilt,),
        removed=(),
        reused=(),
        executed=(),
        executed_targets=(),
    )
    morpher = ModelMorpher(model, path_for)
    result = morpher.apply(snapshot, report, optimizer=optimizer)
    result.refresh_optimizer(optimizer)
    for param in new_alias.parameters():
        optimizer.state[param]["momentum_buffer"] = torch.ones_like(param)
    new_alias_param_ids = {id(param) for param in new_alias.parameters()}

    assert primary_param_ids < optimizer_param_ids(optimizer)
    assert new_alias_param_ids <= optimizer_param_ids(optimizer)

    pass_results = PostPassManager([tying]).apply(model, changed_paths=result.rebuilt_paths, optimizer=optimizer)

    assert pass_results[0].labels() == ("encoders.cam_b",)
    assert model.encoders["cam_b"] is model.encoders["cam_a"]
    assert optimizer_param_ids(optimizer) == {id(param) for param in model.parameters()}
    assert primary_param_ids <= optimizer_param_ids(optimizer)
    assert new_alias_param_ids.isdisjoint(optimizer_param_ids(optimizer))
    assert all(id(param) not in new_alias_param_ids for param in optimizer.state)


def test_low_precision_patch_is_idempotent_and_state_updates() -> None:
    model = Net()
    linear = model.heads["task"]
    true_forward = linear.forward

    first = LowPrecisionPatchPass(LowPrecisionState(tag="a"), module_paths=["heads"]).apply(model)
    assert first.labels() == ("heads.task",)
    assert linear._pysalsa_lp_patched is True
    assert linear._pysalsa_lp_state == LowPrecisionState(tag="a")
    assert same_bound_method(linear._pysalsa_lp_original_forward, true_forward)
    patched_forward = linear.forward

    LowPrecisionPatchPass(LowPrecisionState(tag="b"), module_paths=["heads"]).apply(model)
    assert same_bound_method(linear._pysalsa_lp_original_forward, true_forward)
    assert linear.forward is patched_forward
    assert linear._pysalsa_lp_state == LowPrecisionState(tag="b")

    x = torch.randn(1, 2)
    assert torch.equal(linear(x), true_forward(x))


def test_low_precision_patch_can_restore_original_forward() -> None:
    model = Net()
    linear = model.heads["task"]
    true_forward = linear.forward
    assert "forward" not in vars(linear)

    LowPrecisionPatchPass(LowPrecisionState(tag="a"), module_paths=["heads"]).apply(model)
    assert linear.forward is not true_forward
    assert "forward" in vars(linear)

    LowPrecisionPatchPass(LowPrecisionState(enabled=False, tag="a"), module_paths=["heads"]).apply(model)
    assert same_bound_method(linear.forward, true_forward)
    assert "forward" not in vars(linear)
    assert not hasattr(linear, "_pysalsa_lp_original_forward")
    assert not hasattr(linear, "_pysalsa_lp_state")
    assert not hasattr(linear, "_pysalsa_lp_patched")


def test_low_precision_disabled_state_does_not_patch() -> None:
    model = Net()

    result = LowPrecisionPatchPass(LowPrecisionState(enabled=False), module_paths=["heads"]).apply(model)

    assert result.labels() == ()
    assert not hasattr(model.heads["task"], "_pysalsa_lp_patched")


def test_annotation_refresh_updates_new_module_paths_after_replacement() -> None:
    model = Net()
    AnnotationPass("Net").apply(model)
    old_head = model.heads["task"]
    assert old_head._annotation == "Net.heads.task"

    new_head = torch.nn.Linear(2, 1)
    _set_module(model, ("heads", "task"), new_head)
    AnnotationPass("Net").apply(model)

    assert new_head._annotation == "Net.heads.task"
    assert model.heads._annotation == "Net.heads"


def test_annotation_records_alias_paths_for_tied_modules() -> None:
    model = Net()
    ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]]).apply(model)

    result = AnnotationPass("Net").apply(model)

    assert "encoders.cam_a" in result.labels()
    assert "encoders.cam_b" in result.labels()
    assert model.encoders["cam_a"]._annotation == "Net.encoders.cam_a"
    assert model.encoders["cam_a"]._annotation_paths == (
        "Net.encoders.cam_a",
        "Net.encoders.cam_b",
    )


def test_post_pass_manager_runs_passes_in_order() -> None:
    model = Net()
    manager = PostPassManager(
        [
            ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]]),
            LowPrecisionPatchPass(LowPrecisionState(tag="a"), module_paths=["heads.task"]),
            AnnotationPass("Net"),
        ]
    )

    results = manager.apply(model, changed_paths=["heads.task"])

    assert tuple(result.name for result in results) == ("low_precision", "annotation")
    assert model.encoders["cam_b"] is not model.encoders["cam_a"]
    assert model.heads["task"]._pysalsa_lp_state == LowPrecisionState(tag="a")
    assert model.heads["task"]._annotation == "Net.heads.task"

    all_results = manager.apply(model)
    assert tuple(result.name for result in all_results) == ("parameter_tying", "low_precision", "annotation")
    assert model.encoders["cam_b"] is model.encoders["cam_a"]


def test_post_pass_manager_propagates_touched_alias_paths_to_later_passes() -> None:
    model = Net()
    manager = PostPassManager(
        [
            ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]]),
            LowPrecisionPatchPass(LowPrecisionState(tag="alias"), module_paths=["encoders.cam_b"]),
        ]
    )

    results = manager.apply(model, changed_paths=["encoders.cam_a"])

    assert tuple(result.name for result in results) == ("parameter_tying", "low_precision")
    assert results[0].labels() == ("encoders.cam_b",)
    assert results[1].labels() == ("encoders.cam_b",)
    assert model.encoders["cam_b"] is model.encoders["cam_a"]
    assert model.encoders["cam_a"]._pysalsa_lp_state == LowPrecisionState(tag="alias")


def test_post_pass_manager_skips_irrelevant_path_scoped_passes() -> None:
    model = Net()
    manager = PostPassManager(
        [
            ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]]),
            LowPrecisionPatchPass(LowPrecisionState(tag="a"), module_paths=["heads.task"]),
        ]
    )

    results = manager.apply(model, changed_paths=["unrelated.module"])

    assert results == ()
    assert model.encoders["cam_b"] is not model.encoders["cam_a"]
    assert not hasattr(model.heads["task"], "_pysalsa_lp_patched")
