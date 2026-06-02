from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

from .core import tracked
from .rebuild import ComponentSnapshot, ComponentTarget, RebuildReport


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("pysalsa.pytorch requires PyTorch to be installed") from exc
    return torch


def tensor_equal(left: Any, right: Any) -> bool:
    torch = _torch()
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return False
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and bool(torch.equal(left, right))
    )


def tensor_fingerprint(tensor: Any, *, include_values: bool = False) -> tuple[Any, ...]:
    torch = _torch()
    if not torch.is_tensor(tensor):
        raise TypeError("tensor_fingerprint expects a torch.Tensor")
    result: tuple[Any, ...] = (
        "tensor",
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        bool(tensor.requires_grad),
    )
    if include_values:
        cpu = tensor.detach().cpu().contiguous()
        result += (hash(cpu.numpy().tobytes()),)
    return result


def _type_name(value: Any) -> str:
    typ = type(value)
    return f"{typ.__module__}.{typ.__qualname__}"


def _attribute_fingerprint(value: Any) -> tuple[Any, ...]:
    torch = _torch()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("literal", value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_attribute_fingerprint(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    ((_attribute_fingerprint(key), _attribute_fingerprint(item)) for key, item in value.items()),
                    key=repr,
                )
            ),
        )
    if torch.is_tensor(value):
        return tensor_fingerprint(value)
    if isinstance(value, torch.nn.Module):
        return ("module-ref", _type_name(value))
    if isinstance(value, torch.nn.Parameter):
        return tensor_fingerprint(value)
    return ("repr", _type_name(value), repr(value))


def _module_attributes(module: Any) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    torch = _torch()
    attrs = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if callable(value) or isinstance(value, (torch.nn.Module, torch.nn.Parameter)):
            continue
        attrs.append((name, _attribute_fingerprint(value)))
    return tuple(sorted(attrs))


def module_fingerprint(module: Any, *, include_parameter_values: bool = False) -> tuple[Any, ...]:
    """Return a structural fingerprint for a ``torch.nn.Module``.

    By default this ignores parameter values so a freshly initialized module
    with the same class/tree/parameter shapes is considered equivalent. That is
    the useful mode for preserving an old module and its weights after an
    incremental rebuild proves the structure did not change.
    """

    torch = _torch()
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module_fingerprint expects a torch.nn.Module")

    parameters = []
    for name, parameter in module.named_parameters(recurse=False):
        parameters.append((name, tensor_fingerprint(parameter, include_values=include_parameter_values)))

    buffers = []
    for name, buffer in module.named_buffers(recurse=False):
        buffers.append((name, tensor_fingerprint(buffer, include_values=include_parameter_values)))

    children = []
    for name, child in module.named_children():
        children.append((name, module_fingerprint(child, include_parameter_values=include_parameter_values)))

    return (
        "module",
        _type_name(module),
        module.extra_repr(),
        bool(module.training),
        _module_attributes(module),
        tuple(parameters),
        tuple(buffers),
        tuple(children),
    )


def modules_equivalent(
    left: Any,
    right: Any,
    *,
    include_parameter_values: bool = False,
) -> bool:
    torch = _torch()
    if not isinstance(left, torch.nn.Module) or not isinstance(right, torch.nn.Module):
        return False
    return module_fingerprint(left, include_parameter_values=include_parameter_values) == module_fingerprint(
        right,
        include_parameter_values=include_parameter_values,
    )


def torch_value_equivalent(left: Any, right: Any) -> bool:
    torch = _torch()
    if torch.is_tensor(left) or torch.is_tensor(right):
        return tensor_equal(left, right)
    if isinstance(left, torch.nn.Module) or isinstance(right, torch.nn.Module):
        return modules_equivalent(left, right)
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def tracked_module(
    fn: Callable[..., Any] | None = None,
    *,
    include_parameter_values: bool = False,
):
    """A :func:`pysalsa.tracked` variant for module-building functions."""

    equals = partial(modules_equivalent, include_parameter_values=include_parameter_values)
    if fn is None:
        return tracked(equals=equals, reuse_on_equal=True)
    return tracked(fn, equals=equals, reuse_on_equal=True)


ModulePath = str | tuple[str, ...]
PathResolver = Callable[[ComponentTarget], ModulePath]


@dataclass(frozen=True)
class OptimizerRepair:
    """Parameter changes produced by a model morph."""

    removed_parameter_ids: tuple[int, ...]
    added_parameters: tuple[Any, ...]

    def apply_to(self, optimizer: Any) -> None:
        """Mutate a PyTorch optimizer so it tracks the morphed model."""

        removed = set(self.removed_parameter_ids)
        if removed:
            for group in optimizer.param_groups:
                group["params"] = [param for param in group["params"] if id(param) not in removed]
            for param in list(optimizer.state):
                if id(param) in removed:
                    del optimizer.state[param]

        if not self.added_parameters:
            return

        known = {id(param) for group in optimizer.param_groups for param in group["params"]}
        additions = [param for param in self.added_parameters if id(param) not in known]
        if not additions:
            return

        if optimizer.param_groups:
            optimizer.param_groups[0]["params"].extend(additions)
        else:  # pragma: no cover - torch optimizers normally require params
            optimizer.add_param_group({"params": additions})


@dataclass(frozen=True)
class MorphResult:
    added: tuple[ComponentTarget, ...]
    rebuilt: tuple[ComponentTarget, ...]
    removed: tuple[ComponentTarget, ...]
    reused: tuple[ComponentTarget, ...]
    added_paths: tuple[tuple[str, ...], ...]
    rebuilt_paths: tuple[tuple[str, ...], ...]
    removed_paths: tuple[tuple[str, ...], ...]
    optimizer_repair: OptimizerRepair

    def refresh_optimizer(self, optimizer: Any) -> None:
        self.optimizer_repair.apply_to(optimizer)


def _normalize_path(path: ModulePath) -> tuple[str, ...]:
    if isinstance(path, str):
        if not path:
            raise ValueError("module paths cannot be empty")
        return tuple(path.split("."))
    normalized = tuple(path)
    if not normalized:
        raise ValueError("module paths cannot be empty")
    if not all(isinstance(part, str) and part for part in normalized):
        raise TypeError("module paths must contain non-empty string parts")
    return normalized


def _path_label(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _resolve_child(parent: Any, part: str) -> Any:
    torch = _torch()
    if isinstance(parent, torch.nn.ModuleDict):
        return parent[part]
    return getattr(parent, part)


def _resolve_parent(root: Any, path: tuple[str, ...]) -> tuple[Any, str]:
    parent = root
    for part in path[:-1]:
        parent = _resolve_child(parent, part)
    return parent, path[-1]


def _set_module(root: Any, path: tuple[str, ...], module: Any) -> None:
    torch = _torch()
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"morphed component at {_path_label(path)!r} is not a torch.nn.Module")
    parent, name = _resolve_parent(root, path)
    if isinstance(parent, torch.nn.ModuleDict):
        parent[name] = module
    else:
        setattr(parent, name, module)


def _get_module(root: Any, path: tuple[str, ...]) -> Any | None:
    try:
        return _resolve_child(_resolve_parent(root, path)[0], path[-1])
    except (AttributeError, KeyError):
        return None


def _del_module(root: Any, path: tuple[str, ...]) -> Any | None:
    torch = _torch()
    parent, name = _resolve_parent(root, path)
    old = _get_module(root, path)
    if old is None:
        return None
    if isinstance(parent, torch.nn.ModuleDict):
        del parent[name]
    else:
        delattr(parent, name)
    return old


def _parameters_of(modules: Iterator[Any] | list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    parameters = []
    for module in modules:
        if module is None:
            continue
        parameters.extend(module.parameters())
    return tuple(parameters)


def _assert_morphable(root: Any) -> None:
    typ = type(root)
    name = f"{typ.__module__}.{typ.__qualname__}"
    if "DistributedDataParallel" in name or "FullyShardedDataParallel" in name:
        raise RuntimeError(f"refusing to morph wrapped model {name}")
    if hasattr(root, "_fsdp_wrapped_module"):
        raise RuntimeError("refusing to morph FSDP-wrapped model")


class ModelMorpher:
    """Apply `RebuildPlanner` snapshots to a PyTorch module tree."""

    def __init__(
        self,
        root: Any,
        path_for_target: PathResolver,
        *,
        refresh: Callable[[Any], None] | None = None,
        reject_wrapped: bool = True,
    ) -> None:
        torch = _torch()
        if not isinstance(root, torch.nn.Module):
            raise TypeError("ModelMorpher root must be a torch.nn.Module")
        if reject_wrapped:
            _assert_morphable(root)
        self.root = root
        self.path_for_target = path_for_target
        self.refresh = refresh

    def path(self, target: ComponentTarget) -> tuple[str, ...]:
        return _normalize_path(self.path_for_target(target))

    def install(self, snapshot: ComponentSnapshot) -> MorphResult:
        paths = []
        new_modules = []
        for target, value in snapshot.values.items():
            path = self.path(target)
            _set_module(self.root, path, value)
            paths.append(path)
            new_modules.append(value)
        self._refresh()
        return MorphResult(
            added=tuple(snapshot.values),
            rebuilt=(),
            removed=(),
            reused=(),
            added_paths=tuple(paths),
            rebuilt_paths=(),
            removed_paths=(),
            optimizer_repair=OptimizerRepair((), _parameters_of(new_modules)),
        )

    def apply(self, snapshot: ComponentSnapshot, report: RebuildReport) -> MorphResult:
        removed_modules = []
        added_modules = []
        added_paths = []
        rebuilt_paths = []
        removed_paths = []

        for component in report.removed:
            path = self.path(component)
            removed_modules.append(_del_module(self.root, path))
            removed_paths.append(path)

        for component in report.rebuilt:
            path = self.path(component)
            removed_modules.append(_get_module(self.root, path))
            _set_module(self.root, path, snapshot.values[component])
            added_modules.append(snapshot.values[component])
            rebuilt_paths.append(path)

        for component in report.added:
            path = self.path(component)
            _set_module(self.root, path, snapshot.values[component])
            added_modules.append(snapshot.values[component])
            added_paths.append(path)

        self._refresh()
        removed_parameter_ids = tuple(sorted({id(param) for param in _parameters_of(removed_modules)}))
        return MorphResult(
            added=report.added,
            rebuilt=report.rebuilt,
            removed=report.removed,
            reused=report.reused,
            added_paths=tuple(added_paths),
            rebuilt_paths=tuple(rebuilt_paths),
            removed_paths=tuple(removed_paths),
            optimizer_repair=OptimizerRepair(removed_parameter_ids, _parameters_of(added_modules)),
        )

    def _refresh(self) -> None:
        if self.refresh is not None:
            self.refresh(self.root)


@contextmanager
def preserve_torch_rng() -> Iterator[None]:
    """Run a block without consuming the caller-visible PyTorch RNG state."""

    torch = _torch()
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:  # pragma: no cover - hardware dependent
            torch.cuda.set_rng_state_all(cuda_states)
