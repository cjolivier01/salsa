from __future__ import annotations

from dataclasses import dataclass

from pysalsa import Database, tracked


@dataclass
class BuiltModule:
    kind: str
    shape: tuple[int, ...]


def module_equivalent(left: BuiltModule, right: BuiltModule) -> bool:
    return left.kind == right.kind and left.shape == right.shape


def base_config():
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
        },
        "tasks": {
            "object_detection": {
                "head_config": {"hidden_dim": 256},
                "input_mapping": {"feature_modules.bev_feature_module": "main"},
            },
            "boundary_detection": {
                "head_config": {"hidden_dim": 64},
                "input_mapping": {"feature_modules.boundary_feature_module": "main"},
            },
        },
    }


def make_queries(log: list[str]):
    @tracked(equals=module_equivalent)
    def build_encoder(db: Database, cfg, key: str) -> BuiltModule:
        log.append(f"encoder:{key}")
        run_args = cfg.read("run_args")
        params = cfg.read("encoder_params", default={})
        height, width = params["input_image_shape"]
        return BuiltModule(
            kind=run_args["encoder"],
            shape=(run_args["input-channels"] * 32, height // 4, width // 4),
        )

    @tracked
    def encoder_shape(db: Database, cfg, key: str) -> tuple[int, ...]:
        return build_encoder(db, cfg, key).shape

    @tracked(equals=module_equivalent)
    def build_feature_module(db: Database, cfg, name: str) -> BuiltModule:
        log.append(f"feature:{name}")
        conf = cfg.read(("feature_modules", name, 0))
        required = conf.get("required_modules", [])
        if required:
            in_shape = build_feature_module(db, cfg, required[0]).shape
        else:
            encoder_key = cfg.read("run_args.encoder-key", default="image")
            in_shape = encoder_shape(db, cfg, encoder_key)
        return BuiltModule(conf["type"], (conf.get("out_channels", in_shape[0]), *in_shape[1:]))

    @tracked(equals=module_equivalent)
    def build_head(db: Database, cfg, task_name: str) -> BuiltModule:
        log.append(f"head:{task_name}")
        head_config = cfg.read(("tasks", task_name, "head_config"))
        mapping = cfg.read(("tasks", task_name, "input_mapping"), default={})
        source = next(iter(mapping), None)
        if source and source.startswith("feature_modules."):
            shape = build_feature_module(db, cfg, source.removeprefix("feature_modules.")).shape
        else:
            shape = encoder_shape(db, cfg, cfg.read("run_args.encoder-key", default="image"))
        return BuiltModule(f"Head[{task_name}]", (head_config.get("hidden_dim", shape[0]), *shape))

    return build_encoder, build_feature_module, build_head


def test_one_head_config_change_rebuilds_only_that_head() -> None:
    db = Database()
    cfg = db.config(base_config())
    log: list[str] = []
    _, _, build_head = make_queries(log)

    object_head = build_head(db, cfg, "object_detection")
    boundary_head = build_head(db, cfg, "boundary_detection")

    log.clear()
    next_config = cfg.snapshot()
    next_config["tasks"]["object_detection"]["head_config"]["hidden_dim"] = 512
    assert cfg.set(next_config)

    assert build_head(db, cfg, "object_detection") == BuiltModule(
        "Head[object_detection]",
        (512, 128, 64, 80),
    )
    assert build_head(db, cfg, "object_detection") is not object_head
    assert build_head(db, cfg, "boundary_detection") is boundary_head
    assert log == ["head:object_detection"]


def test_unrelated_task_dataset_change_does_not_rebuild_heads() -> None:
    db = Database()
    cfg = db.config(base_config())
    log: list[str] = []
    _, _, build_head = make_queries(log)

    object_head = build_head(db, cfg, "object_detection")
    boundary_head = build_head(db, cfg, "boundary_detection")

    log.clear()
    next_config = cfg.snapshot()
    next_config["tasks"]["object_detection"]["datasets"] = ["new_dataset"]
    assert cfg.set(next_config)

    assert build_head(db, cfg, "object_detection") is object_head
    assert build_head(db, cfg, "boundary_detection") is boundary_head
    assert log == []


def test_encoder_shape_change_cascades_through_features_and_heads() -> None:
    db = Database()
    cfg = db.config(base_config())
    log: list[str] = []
    _, _, build_head = make_queries(log)

    object_head = build_head(db, cfg, "object_detection")
    boundary_head = build_head(db, cfg, "boundary_detection")

    log.clear()
    next_config = cfg.snapshot()
    next_config["encoder_params"]["input_image_shape"] = [128, 160]
    assert cfg.set(next_config)

    assert build_head(db, cfg, "object_detection") is not object_head
    assert build_head(db, cfg, "boundary_detection") is not boundary_head
    assert log == [
        "encoder:image",
        "feature:bev_feature_module",
        "head:object_detection",
        "feature:boundary_feature_module",
        "head:boundary_detection",
    ]
