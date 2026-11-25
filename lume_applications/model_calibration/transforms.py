from collections.abc import Mapping
from contextlib import nullcontext
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Any

import h5py
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

    def to_archive(self, location: str | Path | h5py.Group) -> None:
        """
        Archive the wrapped model and transforms to an HDF5 file or group.

        Delegates model archiving to the base LUMEModel, then stores transform state.

        Parameters
        ----------
        location : str | Path | h5py.Group
            Path to an HDF5 file (created or overwritten) or an open HDF5 group.
        """
        with (
            h5py.File(location, "w")
            if isinstance(location, (str, Path))
            else nullcontext(location)
        ) as archive:
            # Store model class for restoration
            archive.attrs["model_class"] = (
                f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
            )

            # Check if this is a staged model
            # If so, we need to put the base model in a subgroup to avoid
            # conflicts with transforms_state at the root level
            is_staged = hasattr(self.model, "lume_model_instances")

            if is_staged:
                # Create a "model" subgroup for the staged model
                models_group = archive.create_group("model")
                self.model.to_archive(models_group)
            else:
                # For simple models, archive directly to the root
                self.model.to_archive(archive)

            # Store transform state at root level
            state_bytes = _serialize_dict(self.transforms.state_dict())
            archive.create_dataset(
                "transforms_state",
                data=bytearray(state_bytes),
                dtype="u1",
            )
            archive.attrs["transform_names"] = list(self.transforms.keys())
            archive.attrs["is_staged"] = is_staged

    @classmethod
    def from_archive(
        cls, location: str | Path | h5py.Group
    ) -> "TorchModelWithTransforms":
        """
        Restore a model instance from an archived HDF5 file or group.

        Expects the base model to have its own from_archive method implemented.
        Restores transforms from stored state.

        Parameters
        ----------
        location : str | Path | h5py.Group
            Path to an HDF5 file or an open HDF5 group.

        Returns
        -------
        TorchModelWithTransforms
            The fully reconstructed model with loaded weights and transforms.
        """
        with (
            h5py.File(location, "r")
            if isinstance(location, (str, Path))
            else nullcontext(location)
        ) as archive:
            # Import to get access to the actual model class
            # The base model must implement its own from_archive
            # We'll try to find which model subclass was used and call its from_archive
            import importlib

            # Try to restore using model_class attribute if available
            if "model_class" in archive.attrs:
                model_class_path = archive.attrs["model_class"]
                module_name, class_name = model_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                ModelClass = getattr(module, class_name)

                # Check if this is a staged model - if so, look in the "model" subgroup
                is_staged = archive.attrs.get("is_staged", False)
                if is_staged and "model" in archive:
                    model_archive = archive["model"]
                else:
                    model_archive = archive

                base_model = ModelClass.from_archive(model_archive)
            else:
                raise ValueError(
                    "Archive must contain 'model_class' attribute to identify the model type"
                )

            # Restore transform state
            if "transforms_state" in archive:
                transforms_state = torch.load(
                    BytesIO(archive["transforms_state"][()].tobytes()),
                    map_location="cpu",
                    weights_only=True,
                )
                transform_names = archive.attrs.get("transform_names", [])

                # Recreate transforms from state
                transforms = {}
                for name in transform_names:
                    transform = TorchAffineTransform()
                    prefix = f"{name}."
                    transform_params = {
                        k[len(prefix) :]: v
                        for k, v in transforms_state.items()
                        if k.startswith(prefix)
                    }
                    if transform_params:
                        transform.load_state_dict(transform_params)
                    transforms[name] = transform
            else:
                transforms = {}

        return cls(base_model, transforms)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state into model and transforms.

        Parameters
        ----------
        state_dict : dict
            State dictionary with model and transform states.
        strict : bool
            Passed to underlying nn.Module.load_state_dict()
        """
        super().load_state_dict(state_dict, strict=strict)


def _serialize_dict(data: dict) -> bytes:
    """Serialize a dictionary (for pickle-safe state storage)."""
    buffer = BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()
