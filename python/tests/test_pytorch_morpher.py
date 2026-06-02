from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pysalsa import ComponentGraph, ComponentSnapshot, RebuildPlanner, RebuildReport, target
from pysalsa.pytorch import ModelMorpher, modules_equivalent


class Block(torch.nn.Module):
    def __init__(self, kind: str, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.kind = kind
        self.output_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):  # pragma: no cover - tests focus on structure
        return self.linear(x)


class ToyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoders = torch.nn.ModuleDict()
        self.feature_modules = torch.nn.ModuleDict()
        self.heads = torch.nn.ModuleDict()
        self.cache: tuple[str, ...] = ()

    def rebuild_cache(self) -> None:
        self.cache = tuple(name for name, _ in self.named_modules())


def config() -> dict[str, Any]:
    return {
        "run_args": {"encoder-key": "image", "input-channels": 6},
        "encoder_params": {"input_image_shape": [256, 320]},
        "feature_modules": {
            "bev": [{"type": "Bev", "out_dim": 24}],
            "boundary": [{"type": "Boundary", "required_modules": ["bev"], "out_dim": 12}],
            "unused": [{"type": "Unused", "required_modules": ["bev"], "out_dim": 8}],
        },
        "tasks": {
            "object_detection": {
                "head_config": {"hidden_dim": 16},
                "required_modules": ["bev"],
                "input_mapping": {"feature_modules.bev": "main"},
            },
            "boundary_detection": {
                "head_config": {"hidden_dim": 10},
                "required_modules": ["boundary"],
                "input_mapping": {"feature_modules.boundary": "main"},
            },
        },
    }


def required_features(cfg: dict[str, Any]) -> set[str]:
    keep: set[str] = set()
    stack: list[str] = []
    for task_cfg in cfg["tasks"].values():
        stack.extend(task_cfg.get("required_modules", []))
        for path in task_cfg.get("input_mapping", {}):
            if path.startswith("feature_modules."):
                stack.append(path.removeprefix("feature_modules."))

    while stack:
        name = stack.pop()
        if name in keep:
            continue
        keep.add(name)
        for module_cfg in cfg["feature_modules"].get(name, []):
            stack.extend(module_cfg.get("required_modules", []))
    return keep


def targets(cfg: dict[str, Any]):
    encoder_key = cfg["run_args"].get("encoder-key", "image")
    yield target("encoder", encoder_key)
    keep = required_features(cfg)
    for name in cfg["feature_modules"]:
        if name in keep:
            yield target("feature", name)
    for name in sorted(cfg["tasks"]):
        yield target("head", name)


def graph() -> ComponentGraph:
    component_graph = ComponentGraph()

    @component_graph.component("encoder", equals=modules_equivalent)
    def encoder(ctx, key: str) -> Block:
        run_args = ctx.read("run_args")
        params = ctx.read("encoder_params")
        height, _ = params["input_image_shape"]
        out_dim = run_args["input-channels"] * (height // 128)
        return Block(f"encoder:{key}", run_args["input-channels"], out_dim)

    @component_graph.component("feature", equals=modules_equivalent)
    def feature(ctx, name: str) -> Block:
        module_cfg = ctx.read(("feature_modules", name, 0))
        required = module_cfg.get("required_modules", [])
        if required:
            in_dim = ctx.component("feature", required[0]).output_dim
        else:
            key = ctx.read("run_args.encoder-key", default="image")
            in_dim = ctx.component("encoder", key).output_dim
        return Block(f"feature:{name}", in_dim, module_cfg["out_dim"] + in_dim)

    @component_graph.component("head", equals=modules_equivalent)
    def head(ctx, task_name: str) -> Block:
        task_cfg = ctx.read(("tasks", task_name))
        source = next(iter(task_cfg["input_mapping"]))
        in_dim = ctx.component("feature", source.removeprefix("feature_modules.")).output_dim
        return Block(f"head:{task_name}", in_dim, task_cfg["head_config"]["hidden_dim"])

    return component_graph


def path_for(component):
    name = component.key[0]
    if component.name == "encoder":
        return ("encoders", name)
    if component.name == "feature":
        return ("feature_modules", name)
    if component.name == "head":
        return ("heads", name)
    raise KeyError(component)


def make_model():
    planner = RebuildPlanner(graph(), config(), targets)
    model = ToyNet()
    morpher = ModelMorpher(model, path_for, refresh=lambda root: root.rebuild_cache())
    morpher.install(planner.snapshot)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return planner, model, morpher, optimizer


def optimizer_param_ids(optimizer) -> set[int]:
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def test_install_snapshot_populates_module_tree_and_cache() -> None:
    planner, model, _, _ = make_model()

    assert model.encoders["image"] is planner.snapshot.values[target("encoder", "image")]
    assert sorted(model.feature_modules) == ["bev", "boundary"]
    assert sorted(model.heads) == ["boundary_detection", "object_detection"]
    assert "heads.object_detection.linear" in model.cache


def test_head_only_change_replaces_one_head_and_repairs_optimizer() -> None:
    planner, model, morpher, optimizer = make_model()
    old_encoder = model.encoders["image"]
    old_feature = model.feature_modules["bev"]
    old_boundary_head = model.heads["boundary_detection"]
    old_object_head = model.heads["object_detection"]
    old_head_param_ids = {id(param) for param in old_object_head.parameters()}

    next_config = config()
    next_config["tasks"]["object_detection"]["head_config"]["hidden_dim"] = 32
    report = planner.rebuild(next_config)
    result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
    result.refresh_optimizer(optimizer)

    assert model.encoders["image"] is old_encoder
    assert model.feature_modules["bev"] is old_feature
    assert model.heads["boundary_detection"] is old_boundary_head
    assert model.heads["object_detection"] is not old_object_head
    assert result.rebuilt == (target("head", "object_detection"),)
    assert result.rebuilt_paths == (("heads", "object_detection"),)
    assert old_head_param_ids.isdisjoint(optimizer_param_ids(optimizer))
    assert {id(param) for param in model.heads["object_detection"].parameters()} <= optimizer_param_ids(optimizer)


def test_unrelated_metadata_change_preserves_all_identities() -> None:
    planner, model, morpher, optimizer = make_model()
    identities = {name: id(module) for name, module in model.named_modules()}

    next_config = config()
    next_config["tasks"]["object_detection"]["datasets"] = ["new"]
    report = planner.rebuild(next_config)
    result = morpher.apply(planner.snapshot, report)
    result.refresh_optimizer(optimizer)

    assert result.optimizer_repair.removed_parameter_ids == ()
    assert result.optimizer_repair.added_parameters == ()
    assert {name: id(module) for name, module in model.named_modules()} == identities


def test_encoder_shape_change_replaces_encoder_features_and_heads() -> None:
    planner, model, morpher, _ = make_model()
    old_ids = {name: id(module) for name, module in model.named_modules()}

    next_config = config()
    next_config["encoder_params"]["input_image_shape"] = [128, 160]
    report = planner.rebuild(next_config)
    result = morpher.apply(planner.snapshot, report)

    assert sorted(component.label() for component in result.rebuilt) == [
        "encoder:image",
        "feature:bev",
        "feature:boundary",
        "head:boundary_detection",
        "head:object_detection",
    ]
    assert id(model.encoders["image"]) != old_ids["encoders.image"]
    assert id(model.feature_modules["bev"]) != old_ids["feature_modules.bev"]
    assert id(model.heads["object_detection"]) != old_ids["heads.object_detection"]


def test_added_task_inserts_new_feature_and_head() -> None:
    planner, model, morpher, optimizer = make_model()

    next_config = config()
    next_config["tasks"]["lane_detection"] = {
        "head_config": {"hidden_dim": 7},
        "required_modules": ["unused"],
        "input_mapping": {"feature_modules.unused": "main"},
    }
    report = planner.rebuild(next_config)
    result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
    result.refresh_optimizer(optimizer)

    assert "unused" in model.feature_modules
    assert "lane_detection" in model.heads
    assert sorted(component.label() for component in result.added) == [
        "feature:unused",
        "head:lane_detection",
    ]
    assert {id(param) for param in model.heads["lane_detection"].parameters()} <= optimizer_param_ids(optimizer)


def test_removed_task_deletes_head_and_pruned_feature() -> None:
    planner, model, morpher, optimizer = make_model()
    old_head_param_ids = {id(param) for param in model.heads["boundary_detection"].parameters()}

    next_config = config()
    del next_config["tasks"]["boundary_detection"]
    report = planner.rebuild(next_config)
    for param in model.heads["boundary_detection"].parameters():
        optimizer.state[param]["momentum_buffer"] = torch.ones_like(param)

    result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
    result.refresh_optimizer(optimizer)

    assert "boundary_detection" not in model.heads
    assert "boundary" not in model.feature_modules
    assert sorted(component.label() for component in result.removed) == [
        "feature:boundary",
        "head:boundary_detection",
    ]
    assert old_head_param_ids.isdisjoint(optimizer_param_ids(optimizer))
    assert all(id(param) not in old_head_param_ids for param in optimizer.state)


def test_rejects_distributed_wrapped_models() -> None:
    DistributedDataParallel = type(
        "DistributedDataParallel",
        (torch.nn.Module,),
        {"__init__": lambda self: torch.nn.Module.__init__(self)},
    )

    with pytest.raises(RuntimeError, match="wrapped model"):
        ModelMorpher(DistributedDataParallel(), path_for)


def test_optimizer_repair_preserves_parameter_group_for_rebuilt_module() -> None:
    planner, model, morpher, _ = make_model()
    optimizer = torch.optim.SGD(
        [
            {"params": model.encoders.parameters(), "lr": 0.01},
            {"params": model.heads["object_detection"].parameters(), "lr": 0.2},
            {"params": model.heads["boundary_detection"].parameters(), "lr": 0.3},
        ]
    )

    next_config = config()
    next_config["tasks"]["object_detection"]["head_config"]["hidden_dim"] = 32
    report = planner.rebuild(next_config)
    result = morpher.apply(planner.snapshot, report, optimizer=optimizer)
    result.refresh_optimizer(optimizer)

    new_head_param_ids = {id(param) for param in model.heads["object_detection"].parameters()}
    assert new_head_param_ids <= {id(param) for param in optimizer.param_groups[1]["params"]}
    assert new_head_param_ids.isdisjoint({id(param) for param in optimizer.param_groups[0]["params"]})
    assert optimizer.param_groups[1]["lr"] == 0.2


class SharedParamModule(torch.nn.Module):
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        super().__init__()
        self.weight = parameter


def test_optimizer_repair_preserves_still_live_shared_parameters() -> None:
    shared = torch.nn.Parameter(torch.ones(2, 2))
    root = torch.nn.Module()
    root.mods = torch.nn.ModuleDict(
        {
            "keep": SharedParamModule(shared),
            "drop": SharedParamModule(shared),
        }
    )
    optimizer = torch.optim.SGD(root.parameters(), lr=0.1, momentum=0.9)
    optimizer.state[shared]["momentum_buffer"] = torch.ones_like(shared)

    def shared_path(component):
        return ("mods", component.key[0])

    morpher = ModelMorpher(root, shared_path)
    keep = target("mod", "keep")
    drop = target("mod", "drop")
    report = RebuildReport(
        added=(),
        rebuilt=(),
        removed=(drop,),
        reused=(keep,),
        executed=(),
        executed_targets=(),
    )
    snapshot = ComponentSnapshot(MappingProxyType({keep: root.mods["keep"]}))
    result = morpher.apply(snapshot, report, optimizer=optimizer)
    result.refresh_optimizer(optimizer)

    assert "drop" not in root.mods
    assert shared in optimizer.param_groups[0]["params"]
    assert shared in optimizer.state
    assert result.optimizer_repair.removed_parameter_ids == ()


def test_overlapping_component_paths_are_rejected() -> None:
    root = torch.nn.Module()
    root.parent = torch.nn.Module()

    def nested_path(component):
        if component.name == "parent":
            return ("parent",)
        return ("parent", "child")

    morpher = ModelMorpher(root, nested_path)
    parent = target("parent")
    child = target("child")
    snapshot = ComponentSnapshot(
        MappingProxyType(
            {
                parent: torch.nn.Module(),
                child: torch.nn.Linear(1, 1),
            }
        )
    )

    with pytest.raises(ValueError, match="must not overlap"):
        morpher.install(snapshot)


def test_string_module_paths_reject_empty_segments() -> None:
    root = torch.nn.Module()
    root.heads = torch.nn.ModuleDict()
    morpher = ModelMorpher(root, lambda component: "heads.")
    snapshot = ComponentSnapshot(MappingProxyType({target("head", "x"): torch.nn.Linear(1, 1)}))

    with pytest.raises(ValueError, match="empty segments"):
        morpher.install(snapshot)


def test_rejects_data_parallel_wrapped_models() -> None:
    wrapped = torch.nn.DataParallel(torch.nn.Linear(1, 1))

    with pytest.raises(RuntimeError, match="wrapped model"):
        ModelMorpher(wrapped, path_for)
