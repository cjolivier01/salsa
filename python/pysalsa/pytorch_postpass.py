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

    def apply(self, root: Any, *, changed_paths: Iterable[ModulePath] = ()) -> PostPassResult:
        raise NotImplementedError

    def should_run(self, *, changed_paths: tuple[tuple[str, ...], ...]) -> bool:
        return True


class PostPassManager:
    def __init__(self, passes: Iterable[PostPass] = ()) -> None:
        self.passes = tuple(passes)

    def apply(self, root: Any, *, changed_paths: Iterable[ModulePath] = ()) -> tuple[PostPassResult, ...]:
        normalized = tuple(_normalize_path(path) for path in changed_paths)
        return tuple(
            post_pass.apply(root, changed_paths=normalized)
            for post_pass in self.passes
            if post_pass.should_run(changed_paths=normalized)
        )


class ParameterTyingPass(PostPass):
    name = "parameter_tying"

    def __init__(self, groups: Iterable[Iterable[ModulePath]]) -> None:
        self.groups = tuple(tuple(_normalize_path(path) for path in group) for group in groups)
        for group in self.groups:
            if len(group) < 2:
                raise ValueError("parameter tying groups must include a primary and at least one alias")

    def apply(self, root: Any, *, changed_paths: Iterable[ModulePath] = ()) -> PostPassResult:
        touched: list[tuple[str, ...]] = []
        for group in self.groups:
            primary_path = group[0]
            primary = _get_module(root, primary_path)
            if primary is None:
                raise KeyError(f"primary module path does not exist: {_path_label(primary_path)}")
            for alias_path in group[1:]:
                if _get_module(root, alias_path) is not primary:
                    _set_module(root, alias_path, primary)
                touched.append(alias_path)
        return PostPassResult(self.name, tuple(touched))

    def should_run(self, *, changed_paths: tuple[tuple[str, ...], ...]) -> bool:
        if not changed_paths:
            return True
        tied_paths = tuple(path for group in self.groups for path in group)
        return any(_paths_overlap(changed, tied) for changed in changed_paths for tied in tied_paths)


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
        enabled: bool = True,
    ) -> None:
        self.state = state
        self.enabled = enabled
        self.module_paths = None if module_paths is None else tuple(_normalize_path(path) for path in module_paths)

    def apply(self, root: Any, *, changed_paths: Iterable[ModulePath] = ()) -> PostPassResult:
        torch = _torch()
        touched: list[tuple[str, ...]] = []
        for name, module in root.named_modules():
            path = tuple(name.split(".")) if name else ()
            if not self._matches(path):
                continue
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
            if self.enabled:
                self._patch(module)
            else:
                self._restore(module)
            touched.append(path)
        return PostPassResult(self.name, tuple(touched))

    def _matches(self, path: tuple[str, ...]) -> bool:
        if self.module_paths is None:
            return True
        return any(path == prefix or path[: len(prefix)] == prefix for prefix in self.module_paths)

    def should_run(self, *, changed_paths: tuple[tuple[str, ...], ...]) -> bool:
        if not changed_paths:
            return True
        if self.module_paths is None:
            return True
        return any(_paths_overlap(changed, target) for changed in changed_paths for target in self.module_paths)

    def _patch(self, module: Any) -> None:
        if not hasattr(module, "_pysalsa_lp_original_forward"):
            module._pysalsa_lp_original_forward = module.forward

            def lp_forward(self, *args, **kwargs):
                return self._pysalsa_lp_original_forward(*args, **kwargs)

            module.forward = types.MethodType(lp_forward, module)
        module._pysalsa_lp_state = self.state
        module._pysalsa_lp_patched = True

    def _restore(self, module: Any) -> None:
        if hasattr(module, "_pysalsa_lp_original_forward"):
            module.forward = module._pysalsa_lp_original_forward
            del module._pysalsa_lp_original_forward
        if hasattr(module, "_pysalsa_lp_state"):
            del module._pysalsa_lp_state
        if hasattr(module, "_pysalsa_lp_patched"):
            del module._pysalsa_lp_patched


class AnnotationPass(PostPass):
    name = "annotation"

    def __init__(self, root_name: str) -> None:
        self.root_name = root_name

    def apply(self, root: Any, *, changed_paths: Iterable[ModulePath] = ()) -> PostPassResult:
        touched = []
        for name, module in root.named_modules():
            annotation = self.root_name if not name else f"{self.root_name}.{name}"
            module._annotation = annotation
            touched.append(tuple(name.split(".")) if name else ())
        return PostPassResult(self.name, tuple(touched))


def _paths_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return left == right or left[: len(right)] == right or right[: len(left)] == left
