"""Tests for NeuralNetworkBeamSurrogate."""

import math

import pytest
import torch
from cheetah.particles import ParticleBeam, Species
from lume.variables import ScalarVariable
from lume_cheetah.model import CheetahInitialParticlesMixIn
from lume_cheetah.staged_model import CheetahFinalParticlesMixIn
from lume_torch.variables import TorchScalarVariable
from lume_applications.model_calibration.surrogate_models.beam_surrogate import (
    NeuralNetworkBeamSurrogate,
    NNTwissBeam,
    TwissParameterNetwork,
    InputNormalizer,
)


@pytest.fixture
def simple_input_variables():
    """Create simple input variables (1 scalar) for testing with bounds."""
    return [
        TorchScalarVariable(
            name="Q1_k",
            value_range=(-2.0, 2.0),
            default_value=1.0,
        ),
    ]


class TestTwissParameterNetwork:
    """Test the TwissParameterNetwork class."""

    def test_initialization(self):
        """Test network initialization."""
        network = TwissParameterNetwork(input_size=3, hidden_sizes=[64, 32])
        assert network is not None
        assert len(list(network.parameters())) > 0

    def test_forward_pass(self):
        """Test forward pass through the network."""
        network = TwissParameterNetwork(input_size=3, output_size=7)
        input_data = torch.randn(10, 3)
        output = network(input_data)

        assert output.shape == (10, 7)

    def test_single_sample_forward(self):
        """Test forward pass with single sample."""
        network = TwissParameterNetwork(input_size=3, output_size=7)
        input_data = torch.randn(1, 3)
        output = network(input_data)

        assert output.shape == (1, 7)

    def test_custom_output_size(self):
        """Test network with custom output size."""
        output_size = 10
        network = TwissParameterNetwork(input_size=5, output_size=output_size)
        input_data = torch.randn(2, 5)
        output = network(input_data)

        assert output.shape == (2, output_size)

    def test_gradients_flow(self):
        """Test that gradients flow through the network."""
        network = TwissParameterNetwork(input_size=3, output_size=7)
        input_data = torch.randn(5, 3, requires_grad=True)
        output = network(input_data)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        for param in network.parameters():
            assert param.grad is not None


@pytest.fixture
def twiss_beam():
    network = TwissParameterNetwork(1, hidden_sizes=[8], output_size=6).double()
    torch.nn.init.zeros_(network.network[-1].weight)
    torch.nn.init.zeros_(network.network[-1].bias)
    return NNTwissBeam(
        network=network,
        sigma_tau=1e-3,
        sigma_p=10e-3,
        normalized_emittance_scale=1e-6,
        energy=60e6,
    )


class TestNNTwissBeam:
    @pytest.mark.parametrize("batch_shape", [(), (2, 3)])
    def test_initial_twiss_and_normalized_emittance(self, twiss_beam, batch_shape):
        centroid, covariance, energy = twiss_beam(
            torch.zeros(*batch_shape, 1, dtype=torch.float64)
        )
        assert centroid.shape == (*batch_shape, 6)
        assert covariance.shape == (*batch_shape, 6, 6)
        assert energy.shape == batch_shape
        beam = ParticleBeam.from_distribution(
            mu=centroid,
            cov=covariance,
            energy=energy,
            num_particles=64,
            dtype=torch.float64,
        )
        rest_energy = Species("electron", dtype=torch.float64).mass_eV
        beta_gamma = torch.sqrt((energy / rest_energy).square() - 1)
        for plane in ("x", "y"):
            torch.testing.assert_close(
                getattr(beam, f"beta_{plane}"), torch.ones_like(energy)
            )
            torch.testing.assert_close(
                getattr(beam, f"alpha_{plane}"), torch.zeros_like(energy)
            )
            torch.testing.assert_close(
                getattr(beam, f"emittance_{plane}") * beta_gamma,
                torch.full_like(energy, 1e-6),
                atol=1e-14,
                rtol=1e-10,
            )
        torch.testing.assert_close(beam.sigma_tau, torch.full_like(energy, 1e-3))
        torch.testing.assert_close(beam.sigma_p, torch.full_like(energy, 10e-3))
        torch.testing.assert_close(beam.cov_taup, torch.zeros_like(energy))
        torch.testing.assert_close(centroid, torch.zeros_like(centroid))

    def test_preserves_supplied_network_and_converts_twiss(self):
        network = torch.nn.Linear(1, 6, dtype=torch.float64)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.copy_(
                torch.tensor(
                    [
                        math.log(2.0),
                        0.4,
                        math.log(1.5),
                        math.log(3.0),
                        -0.3,
                        math.log(0.8),
                    ],
                    dtype=torch.float64,
                )
            )
        parameterization = NNTwissBeam(
            network=network,
            sigma_tau=1e-3,
            sigma_p=10e-3,
            normalized_emittance_scale=1e-6,
            energy=60e6,
        )
        centroid, covariance, energy = parameterization(
            torch.zeros(1, dtype=torch.float64)
        )
        beam = ParticleBeam.from_distribution(
            mu=centroid,
            cov=covariance,
            energy=energy,
            num_particles=64,
            dtype=torch.float64,
        )
        beta_gamma = torch.sqrt((energy / beam.species.mass_eV).square() - 1)
        for plane, beta, alpha, normalized_emittance in (
            ("x", 2.0, 0.4, 1.5e-6),
            ("y", 3.0, -0.3, 0.8e-6),
        ):
            torch.testing.assert_close(
                getattr(beam, f"beta_{plane}"), energy.new_tensor(beta)
            )
            torch.testing.assert_close(
                getattr(beam, f"alpha_{plane}"), energy.new_tensor(alpha)
            )
            torch.testing.assert_close(
                getattr(beam, f"emittance_{plane}") * beta_gamma,
                energy.new_tensor(normalized_emittance),
                atol=1e-14,
                rtol=1e-10,
            )
        torch.testing.assert_close(
            covariance[:4, 4:], torch.zeros_like(covariance[:4, 4:])
        )
        torch.testing.assert_close(
            covariance[:2, 2:4], torch.zeros_like(covariance[:2, 2:4])
        )

    def test_surrogate_training_preserves_fixed_moments(
        self, twiss_beam, simple_input_variables
    ):
        with torch.no_grad():
            twiss_beam.network.network[0].weight.fill_(0.2)
            twiss_beam.network.network[0].bias.fill_(0.3)
            twiss_beam.network.network[-1].weight.fill_(0.1)
        model = NeuralNetworkBeamSurrogate(
            simple_input_variables, twiss_beam, num_particles=64, dtype=torch.float64
        )
        inputs = torch.tensor([-1.0, 1.0], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        model.set({"Q1_k": inputs})
        before = model.cheetah_final_particles
        loss = (before.sigma_x.square() + before.sigma_y.square()).sum() * 1e8
        loss.backward()
        assert torch.isfinite(inputs.grad).all() and (inputs.grad != 0).all()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert (
                torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            )
        optimizer.step()
        model.set({"Q1_k": inputs})
        after = model.cheetah_final_particles
        assert not torch.allclose(before.sigma_x, after.sigma_x, atol=0)
        for name in ("sigma_tau", "sigma_p", "cov_taup", "energy"):
            torch.testing.assert_close(getattr(after, name), getattr(before, name))
        model.float()
        converted = model.cheetah_final_particles
        assert converted.particles.dtype == converted.energy.dtype == torch.float32

    @pytest.mark.parametrize(
        "name, value",
        [
            ("sigma_tau", 0.0),
            ("sigma_p", -0.1),
            ("normalized_emittance_scale", float("nan")),
            ("energy", 1e5),
        ],
    )
    def test_invalid_beam_estimates(self, name, value):
        values = dict(
            sigma_tau=1e-3, sigma_p=10e-3, normalized_emittance_scale=1e-6, energy=60e6
        )
        values[name] = value
        with pytest.raises(ValueError, match=name):
            NNTwissBeam(network=torch.nn.Linear(1, 6), **values)

    @pytest.mark.parametrize("output_size", [5, 7])
    def test_requires_six_twiss_outputs(self, twiss_beam, output_size):
        twiss_beam.network = torch.nn.Linear(1, output_size, dtype=torch.float64)
        with pytest.raises(ValueError, match="shape"):
            twiss_beam(torch.zeros(1, dtype=torch.float64))


class MomentNetwork(torch.nn.Module):
    """A coupled, action-dependent distribution with trainable moments and energy."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 13)
        with torch.no_grad():
            self.linear.weight.fill_(0.1)
            self.linear.bias.copy_(torch.linspace(0.1, 1.3, 13))
        self.register_buffer("coupling", torch.ones(6, 6).tril(-1) * 1e-5)

    def forward(self, inputs):
        values = self.linear(inputs)
        centroid = values[..., :6] * 1e-3
        scale = torch.diag_embed(values[..., 6:12].exp() * 1e-4) + self.coupling
        return centroid, scale @ scale.mT, values[..., 12].exp() * 1e8


class TestNeuralNetworkBeamSurrogate:
    """Test the NeuralNetworkBeamSurrogate class."""

    def test_initialization_with_network(self, simple_input_variables):
        """Test initialization with pre-instantiated network."""
        network = MomentNetwork()
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=network,
        )
        assert model.network is network

    def test_initialization_requires_module(self, simple_input_variables):
        """Reject a non-module rather than creating a default Twiss network."""
        with pytest.raises(TypeError, match="torch.nn.Module"):
            NeuralNetworkBeamSurrogate(
                action_variables=simple_input_variables,
                network=None,
            )

    def test_get_network(self, simple_input_variables):
        """Test getting the network."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )
        network = model.get_network()
        assert isinstance(network, MomentNetwork)

    @pytest.mark.parametrize("batch_shape", [(), (1,), (3,), (2, 3)])
    def test_predicted_moments(self, simple_input_variables, batch_shape):
        """Match every centroid/covariance entry, including coupled longitudinal ones."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
            num_particles=64,
            total_charge=1e-12,
            s=1.25,
            species="positron",
            dtype=torch.float64,
        )
        model.set({"Q1_k": torch.full(batch_shape, 0.5, dtype=torch.float64)})
        centroid, covariance, energy = model.network(model._get_input_vector())
        beam = model.cheetah_final_particles
        particles = beam.particles[..., :6]
        centered = particles - particles.mean(dim=-2, keepdim=True)
        assert particles.shape == (*batch_shape, 64, 6)
        torch.testing.assert_close(particles.mean(dim=-2), centroid)
        torch.testing.assert_close(centered.mT @ centered / 63, covariance)
        torch.testing.assert_close(beam.energy, energy)
        torch.testing.assert_close(
            beam.total_charge, torch.tensor(1e-12, dtype=torch.float64)
        )
        torch.testing.assert_close(beam.s, torch.tensor(1.25, dtype=torch.float64))
        assert beam.species.name == "positron"

    def test_final_particles_only_source(self, simple_input_variables):
        """Expose a beam source, not a stage that accepts initial particles."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )

        assert isinstance(model, CheetahFinalParticlesMixIn)
        assert not isinstance(model, CheetahInitialParticlesMixIn)
        assert not hasattr(model, "cheetah_initial_particles")
        beam = model.cheetah_final_particles

        assert isinstance(beam, ParticleBeam)
        assert beam.num_particles == 10000
        torch.testing.assert_close(
            beam.total_charge, beam.species.charge_coulomb * beam.num_particles
        )

    def test_get_input_vector(self, simple_input_variables):
        """Test getting input vector from action variables."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )

        input_vec = model._get_input_vector()

        assert isinstance(input_vec, torch.Tensor)
        assert input_vec.numel() > 0

    def test_custom_parameters(self, simple_input_variables):
        """Test custom beam parameters."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
            num_particles=5000,
            total_charge=torch.tensor(2e-12),
        )

        assert model.num_particles == 5000
        assert model.total_charge is not None

        beam = model.cheetah_final_particles
        assert beam.num_particles == 5000

    def test_device_and_dtype(self, simple_input_variables):
        """Test device and dtype settings."""
        device = torch.device("cpu")
        dtype = torch.float32

        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
            device=device,
            dtype=dtype,
        )

        assert model.device == device
        assert model.dtype == dtype

    def test_supported_variables(self, simple_input_variables):
        """Test supported_variables property."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )

        supported = model.supported_variables
        assert len(supported) == 1
        assert "Q1_k" in supported


class TestGradientFlow:
    """Test gradient flow through the model."""

    def test_gradients_propagate_to_beam(self, simple_input_variables):
        """Test that gradients propagate from beam coordinates to network parameters."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )

        # Get beam (which uses network)
        beam = model.cheetah_final_particles

        # Compute a loss on beam properties
        loss = beam.mu_x.sum()

        # Backward should work
        loss.backward()

        # Check that network parameters have gradients
        for param in model.network.parameters():
            assert param.grad is not None

    def test_input_affects_beam_properties(self, simple_input_variables):
        """Test that changing network weights affects beam properties."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )

        # Get beam with default input
        beam1 = model.cheetah_final_particles
        sigma_x1 = beam1.sigma_x.clone()

        # Manually change network weights significantly
        with torch.no_grad():
            model.network.linear.bias[6].add_(1.0)

        # Get beam with modified network
        beam2 = model.cheetah_final_particles
        sigma_x2 = beam2.sigma_x.clone()

        # Beams should be different (sigma should change)
        assert not torch.allclose(sigma_x1, sigma_x2, rtol=0.1)

    def test_multiple_beams_independent(self, simple_input_variables):
        """Test that multiple beam generations are independent."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )

        beam1 = model.cheetah_final_particles
        beam2 = model.cheetah_final_particles

        # Both should be valid beams
        assert isinstance(beam1, ParticleBeam)
        assert isinstance(beam2, ParticleBeam)

        # They should have same properties (deterministic given network state)
        assert torch.allclose(beam1.mu_x, beam2.mu_x)


class TestInputNormalization:
    """Test input normalization functionality."""

    def test_normalizer_initialization(self, simple_input_variables):
        """Test that normalizer initializes with bounded variables."""
        normalizer = InputNormalizer(simple_input_variables)
        assert normalizer is not None
        assert len(normalizer.bounds) == 1

    def test_normalizer_requires_bounds(self):
        """Test that normalizer requires all variables to have bounds."""
        var = ScalarVariable(
            name="Q1_k",
        )
        # No bounds set
        with pytest.raises(ValueError, match="does not have bounds"):
            InputNormalizer([var])

    def test_normalize_single_value(self, simple_input_variables):
        """Test normalizing a single value."""
        normalizer = InputNormalizer(simple_input_variables)
        # Variable range is (-2.0, 2.0)
        value = torch.tensor(0.0)
        normalized = normalizer.normalize(value)

        # Center of range should normalize to 0
        assert torch.allclose(normalized, torch.tensor(0.0))

    def test_normalize_boundary_values(self, simple_input_variables):
        """Test normalizing boundary values."""
        normalizer = InputNormalizer(simple_input_variables)
        # Variable range is (-2.0, 2.0)

        # Test lower bound
        lower = torch.tensor(-2.0)
        normalized_lower = normalizer.normalize(lower)
        assert torch.allclose(normalized_lower, torch.tensor(-1.0))

        # Test upper bound
        upper = torch.tensor(2.0)
        normalized_upper = normalizer.normalize(upper)
        assert torch.allclose(normalized_upper, torch.tensor(1.0))

    def test_normalize_batch(self, simple_input_variables):
        """Test normalizing a batch of values."""
        normalizer = InputNormalizer(simple_input_variables)
        values = torch.tensor([[-2.0], [0.0], [2.0]])
        normalized = normalizer.normalize(values)

        expected = torch.tensor([[-1.0], [0.0], [1.0]])
        assert torch.allclose(normalized, expected)

    def test_multiple_variables_normalization(self):
        """Test normalization with multiple variables."""
        var1 = ScalarVariable(
            name="var1",
            value_range=(0.0, 10.0),
        )
        var2 = ScalarVariable(
            name="var2",
            value_range=(-5.0, 5.0),
        )

        normalizer = InputNormalizer([var1, var2])

        # Test center points
        values = torch.tensor([[5.0, 0.0]])
        normalized = normalizer.normalize(values)
        expected = torch.tensor([[0.0, 0.0]])
        assert torch.allclose(normalized, expected)

    def test_normalized_input_in_model(self, simple_input_variables):
        """Test that model uses normalized inputs."""
        model = NeuralNetworkBeamSurrogate(
            action_variables=simple_input_variables,
            network=MomentNetwork(),
        )

        input_vec = model._get_input_vector()

        # Normalized input should be in [-1, 1]
        assert torch.all(input_vec >= -1.0) and torch.all(input_vec <= 1.0)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_public_batched_inputs_preserve_beams_and_gradients(
    simple_input_variables, batch_size
):
    network = MomentNetwork()
    model = NeuralNetworkBeamSurrogate(
        simple_input_variables,
        network=network,
        num_particles=64,
        dtype=torch.float64,
    )
    inputs = torch.linspace(
        -1.0, 1.0, batch_size, dtype=torch.float64, requires_grad=True
    )
    rng = torch.random.get_rng_state()
    model.set({"Q1_k": inputs})
    torch.testing.assert_close(model.get(["Q1_k"])["Q1_k"], inputs)
    batched = model.cheetah_final_particles
    assert batched.particles.shape == (batch_size, 64, 7)
    scalar_particles = []
    for value in inputs:
        model.set({"Q1_k": value})
        scalar_particles.append(model.cheetah_final_particles.particles)
    torch.testing.assert_close(batched.particles, torch.stack(scalar_particles))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    assert {id(p) for p in model.parameters()} == {id(p) for p in network.parameters()}
    for _ in range(2):
        model.set({"Q1_k": inputs})
        beam = model.cheetah_final_particles
        loss = (
            beam.mu_x.sum() * 1e3
            + beam.sigma_x.square().sum() * 1e8
            + beam.energy.sum() / 1e8
        )
        loss.backward()
        assert inputs.grad is not None and (inputs.grad != 0).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        assert (network.linear.weight.grad[[0, 6, 12], 0] != 0).all()
        model.zero_grad()
        inputs.grad = None
    model.reset()
    torch.testing.assert_close(
        model.get(["Q1_k"])["Q1_k"], torch.tensor(1.0, dtype=torch.float64)
    )
    model.to(dtype=torch.float32)
    assert model.cheetah_final_particles.particles.dtype == torch.float32


def test_broadcast_distribution(simple_input_variables):
    class BroadcastNetwork(torch.nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_zeros(6),
                torch.eye(6, device=inputs.device, dtype=inputs.dtype) * 1e-8,
                inputs[..., 0].exp() * 1e8,
            )

    model = NeuralNetworkBeamSurrogate(
        simple_input_variables, BroadcastNetwork(), num_particles=64
    )
    model.set({"Q1_k": torch.tensor([-1.0, 0.0, 1.0])})
    beam = model.cheetah_final_particles
    assert beam.particles.shape == (3, 64, 7)
    torch.testing.assert_close(beam.particles[0], beam.particles[2])
    torch.testing.assert_close(beam.energy, torch.tensor([-0.5, 0.0, 0.5]).exp() * 1e8)


@pytest.mark.parametrize(
    "prediction, error, message",
    [
        (torch.zeros(7), TypeError, "must return"),
        ((torch.zeros(6), torch.eye(6), 1e8), TypeError, "as tensors"),
        ((torch.zeros(5), torch.eye(6), torch.tensor(1e8)), ValueError, "centroid"),
        ((torch.zeros(6), torch.eye(5), torch.tensor(1e8)), ValueError, "covariance"),
        (
            (torch.zeros(6), torch.eye(6), torch.tensor(float("nan"))),
            ValueError,
            "finite",
        ),
        ((torch.zeros(6), torch.eye(6), torch.tensor(-1e8)), ValueError, "positive"),
        (
            (
                torch.zeros(6),
                torch.eye(6) + torch.ones(6, 6).tril(-1),
                torch.tensor(1e8),
            ),
            ValueError,
            "symmetric",
        ),
        (
            (torch.zeros(6), -torch.eye(6), torch.tensor(1e8)),
            torch.linalg.LinAlgError,
            "positive-definite",
        ),
    ],
)
def test_invalid_predictions(simple_input_variables, prediction, error, message):
    class InvalidNetwork(torch.nn.Module):
        def forward(self, inputs):
            return prediction

    model = NeuralNetworkBeamSurrogate(
        simple_input_variables, InvalidNetwork(), num_particles=64
    )
    with pytest.raises(error, match=message):
        model.cheetah_final_particles


@pytest.mark.parametrize(
    "variable, message",
    [
        (
            ScalarVariable(name="input", value_range=(-1.0, 1.0)),
            "explicit default value",
        ),
        (
            ScalarVariable(
                name="output",
                read_only=True,
                default_value=0.0,
                value_range=(-1.0, 1.0),
            ),
            "only writable",
        ),
    ],
)
def test_requires_writable_inputs_with_defaults(variable, message):
    with pytest.raises(ValueError, match=message):
        NeuralNetworkBeamSurrogate([variable], MomentNetwork())


@pytest.mark.parametrize(
    "settings",
    [
        {"total_charge": float("nan")},
        {"total_charge": torch.ones(2)},
        {"s": float("inf")},
        {"s": torch.ones(2)},
    ],
)
def test_requires_finite_scalar_beam_settings(simple_input_variables, settings):
    with pytest.raises(ValueError, match="finite scalar"):
        NeuralNetworkBeamSurrogate(simple_input_variables, MomentNetwork(), **settings)
