from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pysalsa import ComponentGraph, RebuildPlanner, target


@dataclass
class BuiltModule:
    kind: str
    shape: tuple[int, ...]


def module_equivalent(left: BuiltModule, right: BuiltModule) -> bool:
    return left.kind == right.kind and left.shape == right.shape


def neural_config() -> dict[str, Any]:
    return {
        "run_args": {
            "encoder": "ImageEncoder",
            "encoder-key": "image",
            "input-channels": 6,
        },
        "encoder_params": {"input_image_shape": [256, 320]},
        "feature_modules": {
            "bev_feature_module": [{"type": "Bev", "out_channels": 128}],
            "boundary_feature_module": [
                {
                    "type": "Boundary",
                    "required_modules": ["bev_feature_module"],
                    "out_channels": 64,
                }
            ],
            "unused_feature_module": [
                {
                    "type": "Unused",
                    "required_modules": ["bev_feature_module"],
                    "out_channels": 32,
                }
            ],
        },
        "tasks": {
            "object_detection": {
                "head_config": {"hidden_dim": 256},
                "required_modules": ["bev_feature_module"],
                "input_mapping": {"feature_modules.bev_feature_module": "main"},
            },
            "boundary_detection": {
                "head_config": {"hidden_dim": 64},
                "required_modules": ["boundary_feature_module"],
                "input_mapping": {"feature_modules.boundary_feature_module": "main"},
            },
        },
    }


def make_neural_graph() -> ComponentGraph:
    graph = ComponentGraph()

    @graph.component("encoder", equals=module_equivalent)
    def encoder(ctx, routing_key: str) -> BuiltModule:
        run_args = ctx.read("run_args")
        params = ctx.read("encoder_params", default={})
        height, width = params["input_image_shape"]
        return BuiltModule(
            kind=f"{run_args['encoder']}[{routing_key}]",
            shape=(run_args["input-channels"] * 32, height // 4, width // 4),
        )

    @graph.component("feature", equals=module_equivalent)
    def feature(ctx, name: str) -> BuiltModule:
        conf = ctx.read(("feature_modules", name, 0))
        required = conf.get("required_modules", [])
        if required:
            in_shape = ctx.component("feature", required[0]).shape
        else:
            encoder_key = ctx.read("run_args.encoder-key", default="image")
            in_shape = ctx.component("encoder", encoder_key).shape
        return BuiltModule(conf["type"], (conf.get("out_channels", in_shape[0]), *in_shape[1:]))

    @graph.component("head", equals=module_equivalent)
    def head(ctx, task_name: str) -> BuiltModule:
        head_config = ctx.read(("tasks", task_name, "head_config"))
        mapping = ctx.read(("tasks", task_name, "input_mapping"), default={})
        source = next(iter(mapping), None)
        if source and source.startswith("feature_modules."):
            in_shape = ctx.component("feature", source.removeprefix("feature_modules.")).shape
        else:
            encoder_key = ctx.read("run_args.encoder-key", default="image")
            in_shape = ctx.component("encoder", encoder_key).shape
        return BuiltModule(f"Head[{task_name}]", (head_config.get("hidden_dim", in_shape[0]), *in_shape))

    return graph


def required_feature_modules(config: dict[str, Any]) -> set[str]:
    modules = config.get("feature_modules", {})
    keep: set[str] = set()
    stack: list[str] = []
    for task_config in config["tasks"].values():
        stack.extend(task_config.get("required_modules", []))
        for path in task_config.get("input_mapping", {}):
            if path.startswith("feature_modules."):
                stack.append(path.removeprefix("feature_modules."))

    while stack:
        name = stack.pop()
        if name in keep:
            continue
        keep.add(name)
        for conf in modules.get(name, []):
            stack.extend(conf.get("required_modules", []))
    return keep


def neural_targets(config: dict[str, Any]):
    encoder_key = config["run_args"].get("encoder-key", "image")
    yield target("encoder", encoder_key)

    keep = required_feature_modules(config)
    for name in config.get("feature_modules", {}):
        if name in keep:
            yield target("feature", name)

    for task_name in sorted(config["tasks"]):
        yield target("head", task_name)


def make_planner() -> tuple[RebuildPlanner, dict[str, Any]]:
    config = neural_config()
    planner = RebuildPlanner(make_neural_graph(), config, neural_targets)
    return planner, config


def test_golden_one_head_hidden_dim_change_rebuilds_only_that_head() -> None:
    planner, config = make_planner()
    next_config = neural_config()
    next_config["tasks"]["object_detection"]["head_config"]["hidden_dim"] = 512

    assert planner.rebuild(next_config).to_golden() == {
        "added": [],
        "rebuilt": ["head:object_detection"],
        "removed": [],
        "reused": [
            "encoder:image",
            "feature:bev_feature_module",
            "feature:boundary_feature_module",
            "head:boundary_detection",
        ],
        "executed": ["head:object_detection"],
        "rebuild_set": ["head:object_detection"],
    }
    assert config["tasks"]["object_detection"]["head_config"]["hidden_dim"] == 256


def test_golden_unrelated_task_metadata_change_rebuilds_nothing() -> None:
    planner, _ = make_planner()
    next_config = neural_config()
    next_config["tasks"]["object_detection"]["datasets"] = ["new_dataset"]

    assert planner.rebuild(next_config).to_golden() == {
        "added": [],
        "rebuilt": [],
        "removed": [],
        "reused": [
            "encoder:image",
            "feature:bev_feature_module",
            "feature:boundary_feature_module",
            "head:boundary_detection",
            "head:object_detection",
        ],
        "executed": [],
        "rebuild_set": [],
    }


def test_golden_encoder_shape_change_cascades_to_features_and_heads() -> None:
    planner, _ = make_planner()
    next_config = neural_config()
    next_config["encoder_params"]["input_image_shape"] = [128, 160]

    assert planner.rebuild(next_config).to_golden() == {
        "added": [],
        "rebuilt": [
            "encoder:image",
            "feature:bev_feature_module",
            "feature:boundary_feature_module",
            "head:boundary_detection",
            "head:object_detection",
        ],
        "removed": [],
        "reused": [],
        "executed": [
            "encoder:image",
            "feature:bev_feature_module",
            "feature:boundary_feature_module",
            "head:boundary_detection",
            "head:object_detection",
        ],
        "rebuild_set": [
            "encoder:image",
            "feature:bev_feature_module",
            "feature:boundary_feature_module",
            "head:boundary_detection",
            "head:object_detection",
        ],
    }


def test_golden_add_task_adds_previously_pruned_feature_and_head() -> None:
    planner, _ = make_planner()
    next_config = neural_config()
    next_config["tasks"]["lane_detection"] = {
        "head_config": {"hidden_dim": 48},
        "required_modules": ["unused_feature_module"],
        "input_mapping": {"feature_modules.unused_feature_module": "main"},
    }

    assert planner.rebuild(next_config).to_golden() == {
        "added": ["feature:unused_feature_module", "head:lane_detection"],
        "rebuilt": [],
        "removed": [],
        "reused": [
            "encoder:image",
            "feature:bev_feature_module",
            "feature:boundary_feature_module",
            "head:boundary_detection",
            "head:object_detection",
        ],
        "executed": ["feature:unused_feature_module", "head:lane_detection"],
        "rebuild_set": ["feature:unused_feature_module", "head:lane_detection"],
    }
