"""Neural network-based beam surrogate model using CheetahInitialParticlesMixIn."""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from cheetah.particles import ParticleBeam, Species
from lume.model import LUMEModel
from lume_cheetah.model import CheetahInitialParticlesMixIn
from lume_cheetah.staged_model import CheetahFinalParticlesMixIn
from lume_cheetah.simulator import CheetahSimulator
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


class TwissParameterNetwork(nn.Module):
    """
    Neural network that maps input variables to Twiss parameters.

    This network takes input variables and outputs Twiss parameters
    (beta_x, alpha_x, log_emittance_x, beta_y, alpha_y, log_emittance_y, energy)
    that can be used to create a Cheetah ParticleBeam via from_twiss.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = None,
        output_size: int = 7,
    ):
        """
        Initialize the Twiss parameter network.

        Parameters
        ----------
        input_size : int
            Number of input variables
        hidden_sizes : list[int], optional
            Sizes of hidden layers. Default: [64, 32]
        output_size : int, optional
            Number of Twiss parameters to output. Default: 7
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)


class NeuralNetworkBeamSurrogate(
    LUMEModel, nn.Module, CheetahInitialParticlesMixIn, CheetahFinalParticlesMixIn
):
    """
    LUMEModel subclass that uses a neural network to generate particle beam distributions.

    This class combines LUMEModel with CheetahInitialParticlesMixIn to provide
    a model that:
    1. Takes input variables (machine parameters)
    2. Passes them through a neural network with trainable parameters
    3. Maps network outputs to Twiss parameters
    4. Creates a Cheetah ParticleBeam using ParticleBeam.from_twiss
    5. Exposes that beam as final particles for a downstream Cheetah stage

    The network parameters are trainable and gradients propagate to the final
    beam coordinates, allowing the network to be optimized for desired beam properties.
    Inputs retain their batch dimensions and autograd graph. A fixed random ensemble
    makes scalar and batched evaluations reproducible. Longitudinal moments, reference
    position, charge and species come from the simulator's initial beam. The generated
    beam has uncoupled transverse planes and zero transverse centroids/dispersion.
    """

    def __init__(
        self,
        simulator: CheetahSimulator,
        action_variables: list[Variable],
        network: Optional[nn.Module] = None,
        input_size: Optional[int] = None,
        hidden_sizes: list[int] = None,
        num_particles: int = 10000,
        total_charge: Optional[torch.Tensor] = None,
        species: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the neural network-based beam surrogate model.

        Parameters
        ----------
        simulator : CheetahSimulator
            Simulator providing reference beam properties and initial action values.
            The surrogate does not track its segment.
        action_variables : list[Variable]
            List of LUME variables for the model's inputs/outputs.
            Writable variables must have bounds and either a default value or a
            simulator-backed getter. Writes condition the network, not the simulator.
        network : nn.Module, optional
            Pre-instantiated neural network. If None, one will be created.
        input_size : int, optional
            Number of input variables. Required if network is None.
        hidden_sizes : list[int], optional
            Sizes of hidden layers in the network. Default: [64, 32]
        num_particles : int, optional
            Number of particles in the generated beam. Default: 10000
        total_charge : torch.Tensor, optional
            Total charge of the beam in Coulombs. Default: None
        species : str, optional
            Particle species (e.g., 'electron'). Default: None
        device : torch.device, optional
            Device for tensors ('cpu' or 'cuda'). Default: None
        dtype : torch.dtype, optional
            Data type for tensors. Default: None

        Raises
        ------
        ValueError
            If any variable is missing bounds required for input normalization
        """
        nn.Module.__init__(self)
        self.simulator = simulator
        self.action_variables = action_variables
        if len({var.name for var in action_variables}) != len(action_variables):
            raise ValueError("Surrogate variable names must be unique.")
        inputs = [var for var in action_variables if not var.read_only]
        if not inputs:
            raise ValueError("The beam surrogate requires at least one writable input.")
        self.input_normalizer = InputNormalizer(inputs)
        if input_size is not None and input_size != len(inputs):
            raise ValueError("input_size must match the number of writable variables.")
        if network is None:
            if input_size is None:
                raise ValueError("Either 'network' or 'input_size' must be provided")
            self.network = TwissParameterNetwork(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
            )
        else:
            self.network = network

        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        self.network.to(device=device, dtype=dtype)
        self.num_particles = num_particles
        reference = simulator.beam_distribution
        self.species = Species(species) if species is not None else reference.species
        for name, value in {
            "total_charge": reference.total_charge
            if total_charge is None
            else total_charge,
            "s": reference.s,
            "sigma_tau": reference.sigma_tau,
            "sigma_p": reference.sigma_p,
            "cov_taup": reference.cov_taup,
            "longitudinal_mean": torch.stack(
                (reference.mu_tau, reference.mu_p), dim=-1
            ),
        }.items():
            self.register_buffer(
                name, value.detach().clone().to(device=device, dtype=dtype)
            )
        self._input_buffers = {}
        for i, var in enumerate(inputs):
            value = var.default_value
            if value is None:
                getter = getattr(var, "_get", None)
                if getter is None:
                    raise ValueError(
                        f"Input '{var.name}' needs a default value or action getter."
                    )
                value = getter(simulator)
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

    def __setattr__(self, name, value):
        if name == "cheetah_initial_particles":
            raise ValueError(
                "Surrogate particles are generated by the network, not assigned."
            )
        super().__setattr__(name, value)

    def _twiss_params_to_beam(
        self,
        twiss_params: torch.Tensor,
    ) -> ParticleBeam:
        """
        Convert network output (Twiss parameters) to a Cheetah ParticleBeam.

        The network outputs emittances in log space to handle the small scales
        (order 1e-8). This method exponentiates them back to real space.

        Parameters
        ----------
        twiss_params : torch.Tensor
            Tensor containing Twiss parameters:
            [beta_x, alpha_x, log(emittance_x), beta_y, alpha_y, log(emittance_y), energy]

        Returns
        -------
        ParticleBeam
            A Cheetah ParticleBeam created from the Twiss parameters
        """
        if twiss_params.ndim == 0 or twiss_params.shape[-1] != 7:
            raise ValueError(
                "The network must output seven Twiss/energy values per sample."
            )
        beta_x, alpha_x, log_emit_x, beta_y, alpha_y, log_emit_y, energy = (
            twiss_params.unbind(-1)
        )
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(0)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(0)
            beam = ParticleBeam.from_twiss(
                beta_x=beta_x.clamp(min=0.1),
                alpha_x=alpha_x,
                emittance_x=log_emit_x.exp(),
                beta_y=beta_y.clamp(min=0.1),
                alpha_y=alpha_y,
                emittance_y=log_emit_y.exp(),
                energy=energy.abs().clamp(min=1e6),
                sigma_tau=self.sigma_tau,
                sigma_p=self.sigma_p,
                cov_taup=self.cov_taup,
                s=self.s,
                num_particles=self.num_particles,
                total_charge=self.total_charge,
                species=self.species,
                device=self.device,
                dtype=self.dtype,
            )
        beam.particles[..., 4:6] += self.longitudinal_mean
        return beam

    def _get_input_vector(self) -> torch.Tensor:
        """
        Get current input vector from action variables and normalize.

        Only includes writable variables (inputs), not read-only variables (outputs).
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
    def cheetah_initial_particles(self) -> ParticleBeam:
        """
        Get the current Cheetah particle beam.

        Uses the neural network to generate a beam based on current input variable
        values. Gradients flow through the network to the beam coordinates.

        Returns
        -------
        ParticleBeam
            A Cheetah ParticleBeam
        """
        return self._twiss_params_to_beam(self.network(self._get_input_vector()))

    @cheetah_initial_particles.setter
    def cheetah_initial_particles(self, val: ParticleBeam) -> None:
        """
        Set the Cheetah particle beam.

        Assignment is rejected: this source generates its beam from network outputs.

        Parameters
        ----------
        val : ParticleBeam
            The particle beam to set
        """
        raise ValueError(
            "Surrogate particles are generated by the network, not assigned."
        )

    @property
    def cheetah_final_particles(self) -> ParticleBeam:
        """Generate a fresh differentiable beam for the next stage."""
        return self.cheetah_initial_particles

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
        return {
            name: (
                getattr(self, self._input_buffers[name])
                if name in self._input_buffers
                else self.supported_variables[name]._get(self.simulator)
            )
            for name in names
        }

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
