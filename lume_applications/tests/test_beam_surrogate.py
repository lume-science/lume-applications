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
    NNBeam,
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


@pytest.fixture(
    params=[
        ("transverse_covariance", 4, 14),
        ("full_covariance", 6, 27),
    ],
    ids=["transverse_covariance", "full_covariance"],
)
def nn_beam(request):
    mode, dimension, output_size = request.param
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, output_size),
    ).double()
    torch.nn.init.zeros_(network[-1].weight)
    torch.nn.init.zeros_(network[-1].bias)
    rest_energy = Species("electron", dtype=torch.float64).mass_eV
    initial_scale = (1e-6 / torch.sqrt((60e6 / rest_energy).square() - 1)).sqrt()
    initial_scales = initial_scale.expand(dimension).clone()
    longitudinal_settings = {}
    if dimension == 4:
        longitudinal_settings = {"sigma_tau": 1e-3, "sigma_p": 10e-3}
    else:
        initial_scales[4:] = torch.tensor([1e-3, 10e-3])
    return NNBeam(
        network=network,
        mode=mode,
        initial_scales=initial_scales,
        energy=60e6,
        **longitudinal_settings,
    )


class TestNNBeam:
    @pytest.mark.parametrize("batch_shape", [(), (2, 3)])
    def test_initial_twiss_and_normalized_emittance(self, nn_beam, batch_shape):
        centroid, covariance, energy = nn_beam(
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

    def test_scale_multipliers_are_learned(self, nn_beam):
        multipliers = torch.linspace(0.7, 1.3, nn_beam.dimension, dtype=torch.float64)
        output = nn_beam.network[-1]
        with torch.no_grad():
            output.bias[nn_beam.factor_size :].copy_(multipliers.log())
        _, covariance, _ = nn_beam(torch.zeros(1, dtype=torch.float64))
        expected_scales = nn_beam.initial_scales * multipliers
        torch.testing.assert_close(
            covariance[: nn_beam.dimension, : nn_beam.dimension],
            torch.diag(expected_scales.square()),
            atol=1e-16,
            rtol=1e-10,
        )
        (
            covariance.diagonal()[: nn_beam.dimension] / nn_beam.initial_scales.square()
        ).sum().backward()
        assert (output.bias.grad[: nn_beam.dimension] != 0).all()
        assert (output.bias.grad[nn_beam.factor_size :] != 0).all()

    def test_surrogate_training_respects_mode(self, nn_beam, simple_input_variables):
        with torch.no_grad():
            nn_beam.network[0].weight.fill_(0.2)
            nn_beam.network[0].bias.fill_(0.3)
            nn_beam.network[-1].weight.fill_(0.1)
        model = NeuralNetworkBeamSurrogate(
            simple_input_variables, nn_beam, num_particles=64, dtype=torch.float64
        )
        inputs = torch.tensor([-1.0, 1.0], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        model.set({"Q1_k": inputs})
        before = model.cheetah_final_particles
        loss = (
            before.particles[..., :4].square().mean() * 1e8
            + before.sigma_tau.square().mean() / 1e-6
            + before.sigma_p.square().mean() / 1e-4
        )
        loss.backward()
        assert torch.isfinite(inputs.grad).all() and (inputs.grad != 0).all()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert (
                torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            )
        assert (nn_beam.network[-1].bias.grad[nn_beam.dimension :] != 0).all()
        optimizer.step()
        model.set({"Q1_k": inputs})
        after = model.cheetah_final_particles
        assert not torch.allclose(before.sigma_x, after.sigma_x, atol=0)
        torch.testing.assert_close(after.energy, before.energy)
        if nn_beam.mode == "full_covariance":
            assert not torch.allclose(before.sigma_tau, after.sigma_tau, atol=0)
            assert not torch.allclose(before.sigma_p, after.sigma_p, atol=0)
        else:
            for name in ("sigma_tau", "sigma_p", "cov_taup"):
                torch.testing.assert_close(getattr(after, name), getattr(before, name))
        model.float()
        converted = model.cheetah_final_particles
        assert converted.particles.dtype == converted.energy.dtype == torch.float32

    @pytest.mark.parametrize(
        "name, value",
        [
            ("sigma_tau", 0.0),
            ("sigma_p", -0.1),
            ("energy", 1e5),
        ],
    )
    def test_invalid_beam_estimates(self, name, value):
        values = dict(
            sigma_tau=1e-3, sigma_p=10e-3, initial_scales=[1e-4] * 4, energy=60e6
        )
        values[name] = value
        with pytest.raises(ValueError, match=name):
            NNBeam(network=torch.nn.Linear(1, 14), **values)

    @pytest.mark.parametrize("output_size", [10, 21])
    def test_requires_mode_specific_outputs(self, nn_beam, output_size):
        nn_beam.network = torch.nn.Linear(1, output_size, dtype=torch.float64)
        with pytest.raises(ValueError, match="shape"):
            nn_beam(torch.zeros(1, dtype=torch.float64))

    @pytest.mark.parametrize("mode", ["unknown", "transverse_twiss"])
    def test_rejects_unknown_mode(self, mode):
        with pytest.raises(ValueError, match="Unknown beam mode"):
            NNBeam(
                network=torch.nn.Linear(1, 14),
                mode=mode,
                sigma_tau=1e-3,
                sigma_p=10e-3,
                initial_scales=[1e-4] * 4,
                energy=60e6,
            )

    @pytest.mark.parametrize("missing", ["sigma_tau", "sigma_p"])
    def test_transverse_mode_requires_longitudinal_widths(self, missing):
        settings = {"sigma_tau": 1e-3, "sigma_p": 10e-3}
        del settings[missing]
        with pytest.raises(ValueError, match=f"{missing} is required"):
            NNBeam(
                network=torch.nn.Linear(1, 14),
                mode="transverse_covariance",
                initial_scales=[1e-4] * 4,
                energy=60e6,
                **settings,
            )

    @pytest.mark.parametrize(
        "scales",
        [[1e-4] * 3, [1e-4, 0.0, 1e-4, 1e-4], [1e-4, float("nan"), 1e-4, 1e-4]],
    )
    def test_requires_positive_scale_vector(self, scales):
        with pytest.raises(ValueError, match="initial_scales"):
            NNBeam(
                network=torch.nn.Linear(1, 14),
                initial_scales=scales,
                energy=60e6,
                sigma_tau=1e-3,
                sigma_p=10e-3,
            )

    def test_full_covariance_rejects_separate_longitudinal_widths(self):
        with pytest.raises(ValueError, match="only accepted in transverse_covariance"):
            NNBeam(
                network=torch.nn.Linear(1, 27),
                mode="full_covariance",
                initial_scales=[1e-4] * 4 + [1e-3, 10e-3],
                energy=60e6,
                sigma_tau=1e-3,
            )

    def test_full_covariance_learns_longitudinal_moments_without_sigma_arguments(self):
        network = torch.nn.Linear(1, 27, dtype=torch.float64)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
            network.bias[20] = 0.2
            network.weight[25, 0] = 0.1
            network.weight[26, 0] = 0.2
            network.weight[20, 0] = 0.1
        parameterization = NNBeam(
            network=network,
            mode="full_covariance",
            initial_scales=[1e-4] * 4 + [1e-3, 10e-3],
            energy=60e6,
        )
        inputs = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
        centroid, covariance, energy = parameterization(inputs)
        beam = ParticleBeam.from_distribution(
            mu=centroid,
            cov=covariance,
            energy=energy,
            num_particles=64,
            dtype=torch.float64,
        )
        torch.testing.assert_close(beam.sigma_tau[0], energy.new_tensor(1e-3))
        torch.testing.assert_close(
            beam.sigma_p[0], energy.new_tensor(math.sqrt((10e-3) ** 2 + (2e-3) ** 2))
        )
        torch.testing.assert_close(beam.cov_taup[0], energy.new_tensor(2e-6))
        (beam.sigma_tau.sum() + beam.sigma_p.sum()).backward()
        assert torch.isfinite(inputs.grad).all() and (inputs.grad != 0).all()
        assert torch.isfinite(network.bias.grad).all()
        assert (network.bias.grad[[20, 25, 26]] != 0).all()
        torch.testing.assert_close(energy, energy.new_full((2,), 60e6))

    @pytest.mark.parametrize(
        "mode, dimension",
        [("transverse_covariance", 4), ("full_covariance", 6)],
    )
    @pytest.mark.parametrize("batch_shape", [(), (2, 3)])
    def test_correlated_covariance_matches_particles(
        self, mode, dimension, batch_shape
    ):
        diagonal = torch.tensor([1.1, 0.9, 1.2, 0.8, 1.3, 0.7], dtype=torch.float64)
        lower = torch.tensor(
            [
                0.2,
                -0.3,
                0.4,
                0.5,
                -0.6,
                0.7,
                0.1,
                0.2,
                -0.1,
                0.3,
                -0.2,
                0.1,
                0.4,
                -0.3,
                0.5,
            ],
            dtype=torch.float64,
        )
        prediction = torch.cat(
            (
                diagonal[:dimension].log(),
                lower[: dimension * (dimension - 1) // 2],
                torch.linspace(0.7, 1.3, dimension, dtype=torch.float64).log(),
            )
        )
        network = torch.nn.Linear(1, prediction.numel(), dtype=torch.float64)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.copy_(prediction)
        initial_scales = torch.tensor(
            [2e-4, 1e-4, 3e-4, 1.5e-4, 1e-3, 10e-3], dtype=torch.float64
        )
        longitudinal_settings = (
            {"sigma_tau": 1e-3, "sigma_p": 10e-3} if dimension == 4 else {}
        )
        parameterization = NNBeam(
            network=network,
            mode=mode,
            initial_scales=initial_scales[:dimension],
            energy=60e6,
            **longitudinal_settings,
        )
        centroid, covariance, energy = parameterization(
            torch.zeros(*batch_shape, 1, dtype=torch.float64)
        )
        scales = initial_scales.clone()
        scales[:dimension] *= torch.linspace(0.7, 1.3, dimension, dtype=torch.float64)
        expected = (
            torch.tensor(
                [
                    [1.21, 0.22, -0.33, 0.55, 0.11, -0.22],
                    [0.22, 0.85, 0.30, -0.44, 0.20, 0.05],
                    [-0.33, 0.30, 1.69, 0.45, -0.07, 0.58],
                    [0.55, -0.44, 0.45, 1.74, 0.10, -0.12],
                    [0.11, 0.20, -0.07, 0.10, 1.84, 0.52],
                    [-0.22, 0.05, 0.58, -0.12, 0.52, 1.04],
                ],
                dtype=torch.float64,
            )
            * scales[:, None]
            * scales[None, :]
        )
        if mode == "transverse_covariance":
            expected[4:, :] = 0
            expected[:, 4:] = 0
            expected[4:, 4:] = torch.diag(scales[4:].square())
        torch.testing.assert_close(
            covariance, expected.expand(*batch_shape, 6, 6), atol=1e-16, rtol=1e-10
        )
        assert (torch.linalg.eigvalsh(covariance) > 0).all()
        beam = ParticleBeam.from_distribution(
            mu=centroid,
            cov=covariance,
            energy=energy,
            num_particles=64,
            dtype=torch.float64,
        )
        centered = beam.particles[..., :6] - beam.particles[..., :6].mean(
            dim=-2, keepdim=True
        )
        torch.testing.assert_close(
            centered.mT @ centered / 63, covariance, atol=1e-16, rtol=1e-10
        )


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
