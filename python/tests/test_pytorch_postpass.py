from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pysalsa.pytorch import _set_module
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


def test_parameter_tying_is_idempotent_and_restores_alias_after_replacement() -> None:
    model = Net()
    pass_ = ParameterTyingPass([["encoders.cam_a", "encoders.cam_b"]])

    result = pass_.apply(model)
    assert result.labels() == ("encoders.cam_b",)
    assert model.encoders["cam_b"] is model.encoders["cam_a"]

    old_primary = model.encoders["cam_a"]
    pass_.apply(model)
    assert model.encoders["cam_b"] is old_primary

    _set_module(model, ("encoders", "cam_b"), torch.nn.Linear(2, 2))
    assert model.encoders["cam_b"] is not old_primary

    pass_.apply(model)
    assert model.encoders["cam_b"] is old_primary


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

    LowPrecisionPatchPass(LowPrecisionState(tag="a"), module_paths=["heads"]).apply(model)
    assert linear.forward is not true_forward

    LowPrecisionPatchPass(LowPrecisionState(tag="a"), module_paths=["heads"], enabled=False).apply(model)
    assert same_bound_method(linear.forward, true_forward)
    assert not hasattr(linear, "_pysalsa_lp_original_forward")
    assert not hasattr(linear, "_pysalsa_lp_state")
    assert not hasattr(linear, "_pysalsa_lp_patched")


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
