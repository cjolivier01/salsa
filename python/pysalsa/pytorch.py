from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

from .core import tracked


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
