"""Tiny YAML-style neural-net build sketch using pysalsa.

This example uses simple Python objects instead of real project classes, but it
models the dependency boundaries from ``net_build_pipeline_for_salsa_designers``:
encoders produce shapes, feature modules depend on selected upstream shapes, and
heads read only the task config plus the mapped input shapes they consume.
"""

from __future__ import annotations

from dataclasses import dataclass

from pysalsa import Database, tracked


@dataclass
class Module:
    kind: str
    shape: tuple[int, ...]


def module_equivalent(left: Module, right: Module) -> bool:
    return left.kind == right.kind and left.shape == right.shape


@tracked(equals=module_equivalent)
def build_encoder(db: Database, cfg, key: str) -> Module:
    run_args = cfg.read("run_args")
    params = cfg.read("encoder_params", default={})
    channels = run_args.get("input-channels", 3)
    height, width = params.get("input_image_shape", [256, 320])
    return Module(kind=run_args.get("encoder", "Encoder"), shape=(channels * 32, height // 4, width // 4))


@tracked
def encoder_shape(db: Database, cfg, key: str) -> tuple[int, ...]:
    return build_encoder(db, cfg, key).shape


@tracked(equals=module_equivalent)
def build_feature_module(db: Database, cfg, name: str) -> Module:
    conf = cfg.read(("feature_modules", name, 0))
    required = conf.get("required_modules", [])
    if required:
        in_shape = build_feature_module(db, cfg, required[0]).shape
    else:
        in_shape = encoder_shape(db, cfg, cfg.read("run_args.encoder-key", default="image"))
    return Module(kind=conf["type"], shape=(conf.get("out_channels", in_shape[0]), *in_shape[1:]))


@tracked(equals=module_equivalent)
def build_head(db: Database, cfg, task_name: str) -> Module:
    task = cfg.read(("tasks", task_name))
    head = task.get("head_config", {})
    mapping = task.get("input_mapping", {})
    source = next(iter(mapping), None)
    if source and source.startswith("feature_modules."):
        shape = build_feature_module(db, cfg, source.removeprefix("feature_modules.")).shape
    else:
        shape = encoder_shape(db, cfg, cfg.read("run_args.encoder-key", default="image"))
    return Module(kind=f"Head[{task_name}]", shape=(head.get("hidden_dim", shape[0]), *shape))


def main() -> None:
    db = Database()
    config = db.config(
        {
            "run_args": {"encoder": "ImageEncoder", "encoder-key": "image", "input-channels": 6},
            "encoder_params": {"input_image_shape": [256, 320]},
            "feature_modules": {
                "bev_feature_module": [{"type": "Bev", "out_channels": 128}],
            },
            "tasks": {
                "object_detection": {
                    "head_config": {"hidden_dim": 256},
                    "input_mapping": {"feature_modules.bev_feature_module": "main"},
                },
                "boundary_detection": {
                    "head_config": {"hidden_dim": 64},
                    "input_mapping": {"feature_modules.bev_feature_module": "main"},
                },
            },
        }
    )

    object_head = build_head(db, config, "object_detection")
    boundary_head = build_head(db, config, "boundary_detection")

    new_config = config.snapshot()
    new_config["tasks"]["object_detection"]["head_config"]["hidden_dim"] = 512
    config.set(new_config)

    assert build_head(db, config, "object_detection") is not object_head
    assert build_head(db, config, "boundary_detection") is boundary_head


if __name__ == "__main__":
    main()
