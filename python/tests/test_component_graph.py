from __future__ import annotations

from dataclasses import dataclass

import pytest

from pysalsa import ComponentGraph, Database


@dataclass
class Component:
    name: str
    width: int


def component_equal(left: Component, right: Component) -> bool:
    return left.name == right.name and left.width == right.width


def test_component_graph_reuses_unchanged_component_values() -> None:
    graph = ComponentGraph()
    calls: list[str] = []

    @graph.component("encoder", equals=component_equal)
    def encoder(ctx):
        calls.append("encoder")
        return Component("encoder", ctx.read("encoder.width"))

    @graph.component("head", equals=component_equal)
    def head(ctx, task_name: str):
        calls.append(f"head:{task_name}")
        encoder_component = ctx.component("encoder")
        width = ctx.read(("tasks", task_name, "hidden"))
        return Component(f"head:{task_name}", encoder_component.width + width)

    db = Database()
    cfg = db.config(
        {
            "encoder": {"width": 10},
            "tasks": {
                "a": {"hidden": 2},
                "b": {"hidden": 4},
            },
        }
    )

    head_a = graph.build(db, cfg, "head", "a")
    head_b = graph.build(db, cfg, "head", "b")

    calls.clear()
    next_cfg = cfg.snapshot()
    next_cfg["tasks"]["a"]["hidden"] = 3
    assert cfg.set(next_cfg)

    assert graph.build(db, cfg, "head", "a") is not head_a
    assert graph.build(db, cfg, "head", "b") is head_b
    assert calls == ["head:a"]


def test_component_graph_rejects_registration_after_use() -> None:
    graph = ComponentGraph()

    @graph.component("value")
    def value(ctx):
        return 1

    db = Database()
    cfg = db.config({})
    assert graph.build(db, cfg, "value") == 1

    with pytest.raises(RuntimeError, match="after a ComponentGraph has been used"):
        graph.register("value", lambda ctx: 2)
