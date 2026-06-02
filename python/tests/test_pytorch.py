from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pysalsa import Database, tracked
from pysalsa.pytorch import modules_equivalent, preserve_torch_rng, tensor_equal


def test_tensor_equal_handles_tensor_comparison() -> None:
    assert tensor_equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
    assert not tensor_equal(torch.tensor([1, 2]), torch.tensor([1, 3]))


def test_tracked_module_reuses_old_module_when_structure_is_equal() -> None:
    db = Database()
    width = db.input(4)
    calls: list[int] = []

    @tracked(equals=modules_equivalent)
    def build_linear(db: Database, width_input):
        calls.append(width_input.get())
        return torch.nn.Linear(width_input.get(), 3)

    first = build_linear(db, width)
    assert width.set(4) is False
    assert build_linear(db, width) is first

    assert width.set(5)
    second = build_linear(db, width)
    assert second is not first

    assert width.set(4)
    third = build_linear(db, width)
    assert third is not second
    assert third.in_features == 4
    assert calls == [4, 5, 4]


def test_preserve_torch_rng_restores_state() -> None:
    torch.manual_seed(1234)
    expected = torch.rand(3)
    torch.manual_seed(1234)
    with preserve_torch_rng():
        _ = torch.rand(100)
    actual = torch.rand(3)
    assert tensor_equal(actual, expected)
