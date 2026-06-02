from __future__ import annotations

import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .pytorch import ModulePath, _get_module, _normalize_path, _path_label, _set_module, _torch


@dataclass(frozen=True)
class PostPassResult:
    name: str
    touched_paths: tuple[tuple[str, ...], ...]

    def labels(self) -> tuple[str, ...]:
        return tuple(_path_label(path) for path in self.touched_paths)


class PostPass:
    name: str

    def apply(
        self,
        root: Any,
        *,
        changed_paths: Iterable[ModulePath] = (),
        removed_paths: Iterable[ModulePath] = (),
    ) -> PostPassResult:
        raise NotImplementedError

    def should_run(
        self,
        *,
        changed_paths: tuple[tuple[str, ...], ...],
        removed_paths: tuple[tuple[str, ...], ...] = (),
    ) -> bool:
        return True


class PostPassManager:
    def __init__(self, passes: Iterable[PostPass] = ()) -> None:
        self.passes = tuple(passes)

    def apply(
        self,
        root: Any,
        *,
        changed_paths: Iterable[ModulePath] = (),
        removed_paths: Iterable[ModulePath] = (),
        optimizer: Any | None = None,
        force: bool = False,
    ) -> tuple[PostPassResult, ...]:
        normalized = tuple(_normalize_path(path) for path in changed_paths)
        removed = tuple(_normalize_path(path) for path in removed_paths)
        full_run = force or (not normalized and not removed)
        active_paths = list(normalized)
        results: list[PostPassResult] = []
        for post_pass in self.passes:
            current_paths = () if full_run else tuple(active_paths)
            if full_run or post_pass.should_run(changed_paths=current_paths, removed_paths=removed):
                result = post_pass.apply(root, changed_paths=current_paths, removed_paths=removed)
                results.append(result)
                if not full_run:
                    active_paths.extend(path for path in result.touched_paths if path not in active_paths)
        if optimizer is not None:
            _prune_optimizer_to_live_parameters(root, optimizer)
        return tuple(results)


class ParameterTyingPass(PostPass):
    name = "parameter_tying"

    def __init__(self, groups: Iterable[Iterable[ModulePath]]) -> None:
        self.groups = tuple(tuple(_normalize_path(path) for path in group) for group in groups)
        for group in self.groups:
            if len(group) < 2:
                raise ValueError("parameter tying groups must include a primary and at least one alias")

    def apply(
        self,
        root: Any,
        *,
        changed_paths: Iterable[ModulePath] = (),
        removed_paths: Iterable[ModulePath] = (),
    ) -> PostPassResult:
        removed = tuple(_normalize_path(path) for path in removed_paths)
        touched: list[tuple[str, ...]] = []
        for group in self.groups:
            primary_path = group[0]
            if _path_is_removed(primary_path, removed):
                continue
            primary = _get_module(root, primary_path)
            if primary is None:
                continue
            for alias_path in group[1:]:
                if _path_is_removed(alias_path, removed):
                    continue
                alias = _get_module(root, alias_path)
                if alias is None:
                    continue
                if alias is not primary:
                    _set_module(root, alias_path, primary)
                    touched.append(alias_path)
        return PostPassResult(self.name, tuple(touched))

    def should_run(
        self,
        *,
        changed_paths: tuple[tuple[str, ...], ...],
        removed_paths: tuple[tuple[str, ...], ...] = (),
    ) -> bool:
        if not changed_paths and not removed_paths:
            return True
        tied_paths = tuple(path for group in self.groups for path in group)
        return any(
            not _path_is_removed(changed, removed_paths) and _paths_overlap(changed, tied)
            for changed in changed_paths
            for tied in tied_paths
        )


@dataclass(frozen=True)
class LowPrecisionState:
    enabled: bool = True
    tag: str = "default"


class LowPrecisionPatchPass(PostPass):
    name = "low_precision"

    def __init__(
        self,
        state: LowPrecisionState | Any,
        *,
        module_paths: Iterable[ModulePath] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.state = state
        self.enabled = getattr(state, "enabled", True) if enabled is None else enabled
        self.module_paths = None if module_paths is None else tuple(_normalize_path(path) for path in module_paths)

    def apply(
        self,
        root: Any,
        *,
        changed_paths: Iterable[ModulePath] = (),
        removed_paths: Iterable[ModulePath] = (),
    ) -> PostPassResult:
        torch = _torch()
        touched: list[tuple[str, ...]] = []
        for name, module in _named_modules(root):
            path = tuple(name.split(".")) if name else ()
            if not self._matches(path):
                continue
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
            changed = False
            if self.enabled:
                changed = self._patch(module)
            else:
                changed = self._restore(module)
            if changed:
                touched.append(path)
        return PostPassResult(self.name, tuple(touched))

    def _matches(self, path: tuple[str, ...]) -> bool:
        if self.module_paths is None:
            return True
        return any(path == prefix or path[: len(prefix)] == prefix for prefix in self.module_paths)

    def should_run(
        self,
        *,
        changed_paths: tuple[tuple[str, ...], ...],
        removed_paths: tuple[tuple[str, ...], ...] = (),
    ) -> bool:
        if not changed_paths and not removed_paths:
            return True
        if self.module_paths is None:
            return True
        candidate_paths = (*changed_paths, *removed_paths)
        return any(_paths_overlap(changed, target) for changed in candidate_paths for target in self.module_paths)

    def _patch(self, module: Any) -> bool:
        changed = False
        if not hasattr(module, "_pysalsa_lp_original_forward"):
            module._pysalsa_lp_had_instance_forward = "forward" in vars(module)
            module._pysalsa_lp_original_forward = module.forward

            def lp_forward(self, *args, **kwargs):
                return self._pysalsa_lp_original_forward(*args, **kwargs)

            module.forward = types.MethodType(lp_forward, module)
            changed = True
        if getattr(module, "_pysalsa_lp_state", None) != self.state:
            changed = True
        module._pysalsa_lp_state = self.state
        if getattr(module, "_pysalsa_lp_patched", None) is not True:
            changed = True
        module._pysalsa_lp_patched = True
        return changed

    def _restore(self, module: Any) -> bool:
        changed = False
        if hasattr(module, "_pysalsa_lp_original_forward"):
            if getattr(module, "_pysalsa_lp_had_instance_forward", False):
                module.forward = module._pysalsa_lp_original_forward
            elif "forward" in vars(module):
                delattr(module, "forward")
            del module._pysalsa_lp_original_forward
            changed = True
        if hasattr(module, "_pysalsa_lp_had_instance_forward"):
            del module._pysalsa_lp_had_instance_forward
            changed = True
        if hasattr(module, "_pysalsa_lp_state"):
            del module._pysalsa_lp_state
            changed = True
        if hasattr(module, "_pysalsa_lp_patched"):
            del module._pysalsa_lp_patched
            changed = True
        return changed


class AnnotationPass(PostPass):
    name = "annotation"

    def __init__(self, root_name: str) -> None:
        self.root_name = root_name

    def apply(
        self,
        root: Any,
        *,
        changed_paths: Iterable[ModulePath] = (),
        removed_paths: Iterable[ModulePath] = (),
    ) -> PostPassResult:
        by_module: dict[int, tuple[Any, list[str], list[tuple[str, ...]]]] = {}
        for name, module in _named_modules(root):
            path = tuple(name.split(".")) if name else ()
            annotation = self.root_name if not name else f"{self.root_name}.{name}"
            _, annotations, paths = by_module.setdefault(id(module), (module, [], []))
            annotations.append(annotation)
            paths.append(path)
        touched = []
        for module, annotations, paths in by_module.values():
            annotation_paths = tuple(annotations)
            if getattr(module, "_annotation", None) != annotations[0]:
                module._annotation = annotations[0]
                touched.extend(paths)
            if getattr(module, "_annotation_paths", None) != annotation_paths:
                module._annotation_paths = annotation_paths
                touched.extend(path for path in paths if path not in touched)
        return PostPassResult(self.name, tuple(touched))


def _paths_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return left == right or left[: len(right)] == right or right[: len(left)] == left


def _path_is_removed(path: tuple[str, ...], removed_paths: Iterable[tuple[str, ...]]) -> bool:
    return any(path == removed or path[: len(removed)] == removed for removed in removed_paths)


def _named_modules(root: Any):
    try:
        yield from root.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility for older PyTorch
        yield from root.named_modules()


def _prune_optimizer_to_live_parameters(root: Any, optimizer: Any) -> None:
    live_parameter_ids = {id(param) for param in root.parameters()}
    for group in optimizer.param_groups:
        group["params"] = [param for param in group["params"] if id(param) in live_parameter_ids]
    for param in list(optimizer.state):
        if id(param) not in live_parameter_ids:
            del optimizer.state[param]
