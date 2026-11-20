"""Neural network-based beam source using CheetahFinalParticlesMixIn."""

import math
from collections.abc import Sequence
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from cheetah.particles import ParticleBeam, Species
from lume.model import LUMEModel
from lume_cheetah.staged_model import CheetahFinalParticlesMixIn
from lume.variables import Variable


class InputNormalizer:
    """Normalizes input values to [-1, 1] range based on variable bounds."""

    def __init__(self, variables: list[Variable]):
        """
        Initialize the input normalizer.

        Parameters
        ----------
        variables : list[Variable]
            List of LUME variables with bounds to normalize
        """
        self.bounds = []
        for var in variables:
            if not hasattr(var, "value_range") or var.value_range is None:
                raise ValueError(
                    f"Variable '{var.name}' does not have bounds defined. "
                    "All variables must have bounds (value_range) for input normalization."
                )
            lower, upper = var.value_range
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError(
                    f"Variable '{var.name}' requires finite increasing bounds."
                )
            self.bounds.append((lower, upper))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalize input values to [-1, 1] range.

        Parameters
        ----------
        values : torch.Tensor
            Input values to normalize, shape (batch_size, num_inputs), (num_inputs,), or scalar

        Returns
        -------
        torch.Tensor
            Normalized values in [-1, 1] range with same shape as input
        """
        if (values.ndim == 0 and len(self.bounds) != 1) or (
            values.ndim > 0 and values.shape[-1] != len(self.bounds)
        ):
            raise ValueError("The final input dimension must match the variable count.")
        bounds = values.new_tensor(self.bounds)
        lower, upper = bounds.unbind(-1)
        return ((values - (lower + upper) / 2) * 2 / (upper - lower)).reshape(
            values.shape
        )


class NNBeam(nn.Module):
    """Neural beam parameterization with zero centroids and fixed reference energy.

    The network must preserve input batch dimensions and return (..., output_size):

    - ``transverse_covariance``: 14 outputs for a 4x4 covariance in (x, px, y, py)
      order: 4 log-diagonal entries, 6 strict-lower-triangular entries, then
      4 log-scale multipliers.
    - ``full_covariance``: 27 outputs for a 6x6 covariance in
      (x, px, y, py, tau, p) order: 6 log-diagonal entries, 15 strict-lower-
      triangular entries, then 6 log-scale multipliers.

    Strict-lower entries fill L in row order: L[1,0], L[2,0], L[2,1], etc.
    Diagonal entries of L are exponentiated. The row scales are
    initial_scales * exp(log_scale_multipliers), and covariance is (D L) @ (D L).T,
    where D is diagonal with those row scales. This gives positive-definite
    covariance for a nonsingular finite factor and supports all correlations.
    initial_scales are standard-deviation scales in coordinate units, not variances.

    Scale multipliers and L are always learned. The user-provided initial_scales
    are fixed buffers.

    Transverse mode requires sigma_tau and sigma_p, fixing the longitudinal
    covariance to diag(sigma_tau**2, sigma_p**2) with zero transverse-longitudinal
    correlations. Full mode takes all six scales through initial_scales and
    does not accept sigma_tau or sigma_p.
    Zero network outputs give covariance diag(initial_scales**2).
    Network weights are not modified.

    Returns (centroid, covariance, energy), with shapes (..., 6), (..., 6, 6)
    and (...), compatible with NeuralNetworkBeamSurrogate.
    """

    def __init__(
        self,
        *,
        network: torch.nn.Module,
        initial_scales: Sequence[float] | torch.Tensor,
        energy: float,
        mode: Literal[
            "transverse_covariance", "full_covariance"
        ] = "transverse_covariance",
        sigma_tau: float | None = None,
        sigma_p: float | None = None,
    ):
        """
        Initialize the selected beam parameterization.

        Parameters
        ----------
        network : torch.nn.Module
            Neural network module predicting distribution parameters.
            Its weights are not modified.
        initial_scales : sequence of float or torch.Tensor
            Positive finite scale vector of length 4 or 6, in coordinate order.
        energy : float
            Total electron energy in eV.
        mode : str, optional
            transverse_covariance (14 outputs, default) or full_covariance
            (27 outputs).
        sigma_tau : float, optional
            Required fixed RMS bunch length in metres for transverse mode.
            Must be omitted in full mode.
        sigma_p : float, optional
            Required fixed relative RMS momentum spread for transverse mode.
            Must be omitted in full mode.
        """

        super().__init__()

        if not isinstance(network, nn.Module):
            raise TypeError("network must be a torch.nn.Module.")
        self.network = network
        dimensions = {
            "transverse_covariance": 4,
            "full_covariance": 6,
        }
        if mode not in dimensions:
            raise ValueError(
                f"Unknown beam mode {mode!r}; choose from {tuple(dimensions)}."
            )
        self.mode = mode
        self.dimension = dimensions[mode]
        self.factor_size = self.dimension * (self.dimension + 1) // 2
        self.output_size = self.factor_size + self.dimension
        scales = torch.as_tensor(initial_scales, dtype=torch.float64).detach().clone()
        if (
            scales.shape != (self.dimension,)
            or not torch.isfinite(scales).all()
            or not (scales > 0).all()
        ):
            raise ValueError(
                f"initial_scales must contain {self.dimension} finite positive values."
            )
        self.register_buffer("initial_scales", scales)
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("energy must be finite and positive.")
        rest_energy = Species("electron", dtype=torch.float64).mass_eV
        if energy <= rest_energy.item():
            raise ValueError("energy must exceed the electron rest energy.")
        self.register_buffer("energy", torch.tensor(energy, dtype=torch.float64))
        longitudinal_covariance: torch.Tensor | None = None
        if mode == "transverse_covariance":
            for name, value in {"sigma_tau": sigma_tau, "sigma_p": sigma_p}.items():
                if value is None:
                    raise ValueError(f"{name} is required for {mode} mode.")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive.")
            longitudinal_covariance = torch.diag(
                torch.tensor([sigma_tau, sigma_p], dtype=torch.float64).square()
            )
        elif sigma_tau is not None or sigma_p is not None:
            raise ValueError(
                "sigma_tau and sigma_p are only accepted in transverse_covariance mode; "
                "provide all six initial_scales for full_covariance."
            )
        self.register_buffer("longitudinal_covariance", longitudinal_covariance)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.network(inputs)
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim == 0
            or values.shape[-1] != self.output_size
        ):
            raise ValueError(
                f"The {self.mode} network must return a tensor with shape "
                f"(..., {self.output_size})."
            )
        covariance = self._cholesky_covariance(values)
        batch_shape = values.shape[:-1]
        if self.longitudinal_covariance is not None:
            transverse_covariance = covariance
            covariance = values.new_zeros(*batch_shape, 6, 6)
            covariance[..., :4, :4] = transverse_covariance
            covariance[..., 4:6, 4:6] = self.longitudinal_covariance
        return (
            covariance.new_zeros(*batch_shape, 6),
            covariance,
            self.energy.expand(batch_shape),
        )

    def _cholesky_covariance(self, values: torch.Tensor) -> torch.Tensor:
        factor = torch.diag_embed(values[..., : self.dimension].exp())
        rows, columns = torch.tril_indices(
            self.dimension, self.dimension, offset=-1, device=values.device
        )
        factor[..., rows, columns] = values[..., self.dimension : self.factor_size]
        scales = self.initial_scales * values[..., self.factor_size :].exp()
        factor = factor * scales.unsqueeze(-1)
        return factor @ factor.mT


class NeuralNetworkBeamSurrogate(LUMEModel, nn.Module, CheetahFinalParticlesMixIn):
    """
    LUMEModel subclass that uses a neural network to generate particle beam distributions from input variables.

    This class combines LUMEModel with CheetahFinalParticlesMixIn to provide
    a model that:
    1. Takes input variables (machine parameters)
    2. Passes them through a neural network with trainable parameters
    3. Predicts a six-dimensional centroid, covariance matrix and reference energy
    4. Creates a Cheetah ParticleBeam using ParticleBeam.from_distribution
    5. Exposes that beam as final particles for a downstream Cheetah stage

    The network parameters are trainable and gradients propagate to the final
    beam coordinates, allowing the network to be optimized for desired beam properties.
    Inputs retain their batch dimensions and autograd graph. A fixed random ensemble
    makes scalar and batched evaluations reproducible. Reference position, charge
    and species are explicit constructor settings. All six-dimensional moments,
    including cross-plane correlations and longitudinal moments, come from the network.
    """

    def __init__(
        self,
        action_variables: list[Variable],
        network: nn.Module,
        *,
        num_particles: int = 10000,
        total_charge: float | torch.Tensor | None = None,
        s: float | torch.Tensor = 0.0,
        species: str = "electron",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the neural network-based beam surrogate model.

        Parameters
        ----------
        action_variables : list[Variable]
            Writable LUME input variables with bounds and explicit default values.
            Writes condition the network. Read-only variables are not supported;
            downstream diagnostics belong to the downstream model. No simulator
            or action getter is used. Use TorchScalarVariable for tensor-valued
            or batched inputs.
        network : nn.Module
            Module called with normalized writable action values of shape
            (..., num_inputs), in action_variables order. Must return a tuple
            (centroid, covariance, energy) of tensors with shapes (..., 6),
            (..., 6, 6), and (...) and broadcastable batch dimensions.
            Coordinates are Cheetah's (x, px, y, py, tau, p); energy is in eV.
            The module must produce finite moments, a symmetric positive-definite
            covariance and positive energy. No clamping or moment transforms are
            applied. Parameters and buffers are moved to device and dtype.
        num_particles : int, optional
            Number of particles in the generated beam. Default: 10000
        total_charge : float or torch.Tensor, optional
            Scalar total charge in Coulombs. If None, use num_particles times
            the charge of one particle of the requested species.
        s : float or torch.Tensor, optional
            Scalar reference position along the beamline in metres. Default: 0.
        species : str, optional
            Particle species. Default: 'electron'.
        device : torch.device, optional
            Device for tensors ('cpu' or 'cuda'). Default: None
        dtype : torch.dtype, optional
            Data type for tensors. Default: None

        Raises
        ------
        ValueError
            If inputs are read-only, lack bounds or default values, or beam
            settings are not finite scalars.
        """
        nn.Module.__init__(self)
        self.action_variables = action_variables
        if len({var.name for var in action_variables}) != len(action_variables):
            raise ValueError("Surrogate variable names must be unique.")
        if any(var.read_only for var in action_variables):
            raise ValueError(
                "The beam surrogate supports only writable input variables."
            )
        inputs = action_variables
        if not inputs:
            raise ValueError("The beam surrogate requires at least one writable input.")
        self.input_normalizer = InputNormalizer(inputs)
        if not isinstance(network, nn.Module):
            raise TypeError("network must be a torch.nn.Module.")
        self.network = network

        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        self.network.to(device=device, dtype=dtype)
        self.num_particles = num_particles
        self.species = Species(species, device=device, dtype=dtype)
        for name, value in {
            "total_charge": self.species.charge_coulomb * num_particles
            if total_charge is None
            else total_charge,
            "s": s,
        }.items():
            tensor = torch.as_tensor(value, device=device, dtype=dtype).detach().clone()
            if tensor.ndim != 0 or not torch.isfinite(tensor):
                raise ValueError(f"{name} must be a finite scalar.")
            self.register_buffer(name, tensor)
        self._input_buffers = {}
        for i, var in enumerate(inputs):
            value = var.default_value
            if value is None:
                raise ValueError(f"Input '{var.name}' needs an explicit default value.")
            tensor = torch.as_tensor(value, device=device, dtype=dtype).detach().clone()
            self.register_buffer(f"_default_{i}", tensor.clone())
            self.register_buffer(f"_input_{i}", tensor, persistent=False)
            self._input_buffers[var.name] = f"_input_{i}"

    @property
    def device(self) -> torch.device:
        return self.s.device

    @property
    def dtype(self) -> torch.dtype:
        return self.s.dtype

    def _distribution_to_beam(
        self,
        centroid: torch.Tensor,
        covariance: torch.Tensor,
        energy: torch.Tensor,
    ) -> ParticleBeam:
        """Create a reproducible particle ensemble from predicted 6D moments."""
        if centroid.ndim < 1 or centroid.shape[-1] != 6:
            raise ValueError("The centroid must have shape (..., 6).")
        if covariance.ndim < 2 or covariance.shape[-2:] != (6, 6):
            raise ValueError("The covariance must have shape (..., 6, 6).")
        if not all(
            torch.isfinite(value).all() for value in (centroid, covariance, energy)
        ):
            raise ValueError(
                "Predicted centroid, covariance and energy must be finite."
            )
        if not torch.allclose(covariance, covariance.mT, atol=0.0, rtol=1e-5):
            raise ValueError("The covariance must be symmetric.")
        if not (energy > 0).all():
            raise ValueError("The reference energy must be positive.")
        batch_shape = torch.broadcast_shapes(
            centroid.shape[:-1], covariance.shape[:-2], energy.shape
        )
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(0)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(0)
            return ParticleBeam.from_distribution(
                mu=centroid.expand(*batch_shape, 6),
                cov=covariance.expand(*batch_shape, 6, 6),
                energy=energy.expand(batch_shape),
                s=self.s,
                num_particles=self.num_particles,
                total_charge=self.total_charge,
                species=self.species,
                device=self.device,
                dtype=self.dtype,
            )

    def _get_input_vector(self) -> torch.Tensor:
        """
        Get current input vector from action variables and normalize.

        Includes all configured writable variables in action_variables order.
        Returns normalized values in [-1, 1] range based on variable bounds.

        Returns
        -------
        torch.Tensor
            Normalized input vector for the neural network
        """
        values = torch.broadcast_tensors(
            *(getattr(self, name) for name in self._input_buffers.values())
        )
        return self.input_normalizer.normalize(torch.stack(values, dim=-1))

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
        """
        Generate a fresh differentiable Cheetah particle beam for the next stage.

        Uses the neural network to generate a beam based on current input variable
        values. Gradients flow through the network to the beam coordinates.

        Returns
        -------
        ParticleBeam
            A Cheetah ParticleBeam
        """
        prediction = self.network(self._get_input_vector())
        if (
            not isinstance(prediction, tuple)
            or len(prediction) != 3
            or not all(isinstance(value, torch.Tensor) for value in prediction)
        ):
            raise TypeError(
                "The network must return (centroid, covariance, energy) as tensors."
            )
        return self._distribution_to_beam(*prediction)

    def get_network(self) -> nn.Module:
        """
        Get the neural network.

        Returns
        -------
        nn.Module
            The underlying neural network with trainable parameters
        """
        return self.network

    # Abstract method implementations from LUMEModel

    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Get values of specified variables.

        Parameters
        ----------
        names : list[str]
            List of variable names to get

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their values
        """
        return {name: getattr(self, self._input_buffers[name]) for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        """
        Set values of specified variables.

        Parameters
        ----------
        values : dict[str, Any]
            Dictionary mapping variable names to their values
        """
        for name, value in values.items():
            setattr(
                self,
                self._input_buffers[name],
                torch.as_tensor(value, device=self.device, dtype=self.dtype),
            )

    def reset(self) -> None:
        """Restore initial PV inputs without resetting fitted network parameters."""
        for i, buffer in enumerate(self._input_buffers.values()):
            setattr(self, buffer, getattr(self, f"_default_{i}").clone())

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """
        Get the variable mapping used by LUME validation and staged routing.

        Returns
        -------
        dict[str, Variable]
            Variables keyed by name
        """
        return {var.name: var for var in self.action_variables}
