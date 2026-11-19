from collections.abc import Mapping
from itertools import chain
from typing import Any

import torch
from lume.model import LUMEModel
from lume.variables import Variable

from .gradient_descent import get_trainable_parameters
from .utils import (
    BidirectionalTransform,
    validate_affine_parameters,
    validate_model_transforms,
    validate_torch_transforms,
    validate_transform_module,
    validate_transform_parameters,
)


class AffineTransform:
    """Backend-independent affine calibration with finite scalar coefficients.

    transform(value) = scale * value + offset; untransform(value) is the inverse.
    Scale may be positive or negative, but not zero.

    Parameters
    ----------
    scale : float
        Initial scale value.
    offset : float
        Initial offset value.

    """

    def __init__(self, scale: float = 1.0, offset: float = 0.0) -> None:
        validate_affine_parameters(scale, offset)
        self.scale: float | torch.Tensor = float(scale)
        self.offset: float | torch.Tensor = float(offset)

    def _validate_parameters(self) -> None:
        validate_affine_parameters(float(self.scale), float(self.offset))

    def transform(self, value: Any) -> Any:
        self._validate_parameters()
        return self.scale * value + self.offset

    def untransform(self, value: Any) -> Any:
        self._validate_parameters()
        return (value - self.offset) / self.scale

    def get_transform_params(self) -> dict[str, float]:
        """Return scalar coefficients for reporting."""
        return {"scale": float(self.scale), "offset": float(self.offset)}

    def reset_to_identity(self) -> None:
        self.scale = 1.0
        self.offset = 0.0


class TorchAffineTransform(AffineTransform, torch.nn.Module):
    """Invertible affine calibration with a signed learnable scale.

    transform(value) = scale * value + offset; untransform(value) is the inverse.
    forward(value) delegates to transform(value) for normal Torch module calls.

    Parameters
    ----------
    scale : float
        Initial scale value.
    offset : float
        Initial offset value.
    dtype : torch.dtype | None, optional
        The data type for the parameters. Defaults to None, which uses the default dtype.
    device : torch.device | str | None, optional
        The device for the parameters. Defaults to None.

    """

    def __init__(
        self,
        scale: float = 1.0,
        offset: float = 0.0,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        values = torch.tensor(
            [scale, offset],
            dtype=torch.get_default_dtype() if dtype is None else dtype,
            device=device,
        )
        validate_transform_parameters(values)
        self.scale = torch.nn.Parameter(values[0].clone())
        self.offset = torch.nn.Parameter(values[1].clone())

    def _validate_parameters(self) -> None:
        validate_transform_parameters(torch.stack((self.scale, self.offset)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.transform(value)

    def get_transform_params(self) -> dict[str, float]:
        """Return detached parameter values for reporting."""
        return {"scale": self.scale.item(), "offset": self.offset.item()}

    def reset_to_identity(self) -> None:
        """Reset values without replacing optimizer parameter references."""
        with torch.no_grad():
            self.scale.fill_(1.0)
            self.offset.zero_()


class ModelWithTransforms(LUMEModel):
    """Apply bidirectional transforms at a LUME model's public set/get boundary.

    Pass a mapping of names to transform instances. The wrapper owns a copy of
    the mapping; the underlying model is not modified. Each instance must
    implement transform(value) and untransform(value).

    set() applies transform; get() applies untransform to configured inputs
    and read-only outputs. Unconfigured values pass through unchanged. The base
    model validates values in its native coordinates and enforces read-only
    status; supported_variables retains that native-coordinate metadata.
    Training inputs and targets must use the wrapper's external coordinates.
    This adapter implements LUMEModel; simulator-specific APIs remain accessible
    through model.
    reset() resets the underlying simulator, preserving fitted transforms.
    It makes no assumptions about Torch or differentiability.

    Parameters
    ----------
    model : LUMEModel
        The underlying LUME model to wrap.
    transforms : Mapping[str, BidirectionalTransform]
        A mapping of variable names to transform instances.

    """

    def __init__(
        self,
        model: LUMEModel,
        transforms: Mapping[str, BidirectionalTransform],
    ) -> None:
        validate_model_transforms(model, transforms)
        self.model = model
        self.transforms: Mapping[str, BidirectionalTransform] | torch.nn.ModuleDict = (
            dict(transforms)
        )

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self.model.supported_variables

    def set(self, values: dict[str, Any]) -> None:
        """Transform supplied inputs once, then let the base model validate/set."""
        # Variable limits describe effective values, not uncalibrated commands.
        self._set(values)

    def _set(self, values: dict[str, Any]) -> None:
        transforms = self.transforms
        self.model.set(
            {
                name: transforms[name].transform(value) if name in transforms else value
                for name, value in values.items()
            }
        )

    def get(self, names: list[str]) -> dict[str, Any]:
        return self._get(names)

    def _get(self, names: list[str]) -> dict[str, Any]:
        values = self.model.get(names)
        transforms = self.transforms
        return {
            name: transforms[name].untransform(value) if name in transforms else value
            for name, value in values.items()
        }

    def reset(self) -> None:
        self.model.reset()


class TorchModelWithTransforms(ModelWithTransforms, torch.nn.Module):
    """Torch adapter for a differentiable set/get model.

    Pass a LUMEModel/nn.Module and a nonempty mapping of names to bidirectional
    nn.Module instances. Initialization registers them as self.transforms,
    matching the model's first floating-point parameter/buffer's dtype/device.
    The underlying model's module and parameter registries are not modified.

    Registering the base model and transforms as separate children includes both
    simulator and transform parameters in parameters(), state_dict(), and to().
    transform_parameters exposes only trainable transform parameters for a
    separate optimizer group.
    No parameter-list override is needed, and shared parameters appear once.

    This preserves autograd; the base model and transforms must themselves
    preserve tensor graphs. Always call this wrapper's set(), not model.set(),
    and set fresh inputs after optimizer updates to rebuild the graph.

    Parameters
    ----------
    model : LUMEModel
        The base model to be wrapped.
    transforms : Mapping[str, BidirectionalTransform]
        A mapping of variable names to their corresponding bidirectional transforms.

    """

    def __init__(
        self,
        model: LUMEModel,
        transforms: Mapping[str, BidirectionalTransform],
    ) -> None:
        validate_model_transforms(model, transforms)
        torch_model = validate_torch_transforms(model)
        if not transforms:
            raise ValueError("transforms must be nonempty")
        modules = torch.nn.ModuleDict()
        module_ids: set[int] = set()
        for name, transform in transforms.items():
            module = validate_transform_module(transform, module_ids)
            module_ids.add(id(module))
            modules[name] = module
        reference = next(
            (
                value
                for value in chain(torch_model.parameters(), torch_model.buffers())
                if value.is_floating_point()
            ),
            None,
        )
        if reference is not None:
            modules.to(device=reference.device, dtype=reference.dtype)
        torch.nn.Module.__init__(self)
        ModelWithTransforms.__init__(self, model, transforms)
        self.transforms: torch.nn.ModuleDict = modules

    @property
    def transform_parameters(self) -> list[torch.nn.Parameter]:
        """Return unique trainable transform parameters, excluding simulator parameters."""
        return get_trainable_parameters(self.transforms)
