from collections.abc import Iterable, Mapping, Sequence
import math
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import torch
from lume.model import LUMEModel


@runtime_checkable
class BidirectionalTransform(Protocol):
    """Map external values to model values and back using BoTorch-style names."""

    def transform(self, value: Any) -> Any: ...

    def untransform(self, value: Any) -> Any: ...


def validate_positive_integer(value: int, name: str) -> None:
    """Require a positive integer, excluding booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_tensor_batch(
    values: Iterable[torch.Tensor],
    *,
    name: str,
    dimension: str = "sample",
) -> int:
    """Validate a common nonzero leading dimension and return its size."""
    values = list(values)
    if not values:
        raise ValueError(f"{name} tensors must be nonempty")
    if any(not isinstance(value, torch.Tensor) or value.ndim == 0 for value in values):
        raise ValueError(f"{name} tensors must have a {dimension} dimension")
    count = values[0].shape[0]
    if count == 0 or any(value.shape[0] != count for value in values):
        raise ValueError(f"{name} {dimension} counts must match and be nonzero")
    return count


def validate_finite_tensors(values: Iterable[torch.Tensor], *, name: str) -> None:
    """Check finiteness after tensor types and dimensions have been validated."""
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError(f"{name} must be finite")


def validate_training_data(
    inputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    require_finite: bool = False,
) -> int:
    """Validate paired training mappings, optionally requiring finite data."""
    if not inputs or not targets:
        raise ValueError("inputs and targets must both be nonempty")
    data = [*inputs.values(), *targets.values()]
    count = validate_tensor_batch(data, name="input and target")
    if require_finite:
        validate_finite_tensors(data, name="inputs and targets")
    return count


def validate_loss_weights(
    loss_weights: Mapping[str, float] | None,
    targets: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Validate target weights and return a copy with omitted weights set to one."""
    weights = {} if loss_weights is None else dict(loss_weights)
    if weights.keys() - targets.keys():
        raise ValueError("loss_weights contains unknown target names")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights.values()
    ):
        raise ValueError("loss_weights must be finite nonnegative numbers")
    validated_weights = {name: float(weights.get(name, 1.0)) for name in targets}
    if not any(validated_weights.values()):
        raise ValueError("at least one loss weight must be positive")
    return validated_weights


def validate_prediction_shape(
    prediction: torch.Tensor,
    target: torch.Tensor,
    name: str,
) -> None:
    """Require a tensor prediction with exactly the target shape, not broadcasting."""
    if not isinstance(prediction, torch.Tensor) or prediction.shape != target.shape:
        raise ValueError(
            f"prediction for {name!r} must match target shape {target.shape}"
        )


def clone_tensor_mapping(
    values: Mapping[str, torch.Tensor],
    *,
    limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Detach and copy tensors, optionally selecting the first leading entries."""
    return {
        name: (value if limit is None else value[:limit]).detach().clone()
        for name, value in values.items()
    }


def validate_affine_parameters(scale: float, offset: float) -> None:
    """Require finite scalar affine coefficients and an invertible scale."""
    if not isinstance(scale, Real) or not isinstance(offset, Real):
        raise TypeError("scale and offset must be real scalars")
    if not math.isfinite(scale) or not math.isfinite(offset):
        raise ValueError("scale and offset must be finite scalars")
    if scale == 0:
        raise ValueError("scale must be nonzero")


def validate_transform_parameters(values: torch.Tensor) -> None:
    """Require finite Torch affine parameters and a nonzero scale."""
    if not values.is_floating_point():
        raise TypeError("transform parameters must have a floating-point dtype")
    if values.shape != (2,) or not torch.isfinite(values).all():
        raise ValueError("scale and offset must be finite scalars")
    if values[0] == 0:
        raise ValueError("scale must be nonzero")


def validate_transform_names(model: LUMEModel, names: Sequence[str]) -> None:
    """Require transform names to identify supported inputs or outputs."""
    supported = model.supported_variables
    for name in names:
        if name not in supported:
            raise ValueError(f"unknown transform variable: {name!r}")


def validate_bidirectional_transform(
    transform: object, name: str
) -> BidirectionalTransform:
    """Require an instance with callable transform and untransform methods."""
    if (
        isinstance(transform, type)
        or not isinstance(transform, BidirectionalTransform)
        or not callable(transform.transform)
        or not callable(transform.untransform)
    ):
        raise TypeError(
            f"transform for {name!r} must be an instance with callable "
            "transform and untransform methods"
        )
    return transform


def validate_model_transforms(
    model: LUMEModel,
    transforms: Mapping[str, BidirectionalTransform],
) -> None:
    """Validate wrapper transforms without modifying the model."""
    if not isinstance(model, LUMEModel):
        raise TypeError("model must be a LUMEModel")
    if not isinstance(transforms, (Mapping, torch.nn.ModuleDict)):
        raise TypeError("provide a mapping of variable names to transforms")
    for name, transform in transforms.items():
        if not isinstance(name, str) or not name:
            raise ValueError("transform names must be nonempty strings")
        validate_bidirectional_transform(transform, name)
    validate_transform_names(model, list(transforms))


def validate_torch_transforms(model: LUMEModel) -> torch.nn.Module:
    """Require and return a Torch-compatible underlying model."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("TorchModelWithTransforms requires an nn.Module")
    return model


def validate_transform_module(
    transform: object, module_ids: set[int]
) -> torch.nn.Module:
    """Require an independent Torch transform module."""
    if not isinstance(transform, torch.nn.Module):
        raise TypeError("Torch transforms must be nn.Module instances")
    validate_bidirectional_transform(transform, "Torch transform")
    if id(transform) in module_ids:
        raise ValueError("Torch transforms must be independent modules")
    return transform
