"""Tests for NeuralNetworkBeamSurrogate."""

import pytest
import torch
from cheetah.accelerator import Drift, Quadrupole, Screen, Segment
from cheetah.particles import ParticleBeam
from lume_cheetah.actions import (
    CheetahWritableScalarVariable,
)
from lume_cheetah.simulator import CheetahSimulator
from lume_applications.model_calibration.surrogate_models.beam_surrogate import (
    NeuralNetworkBeamSurrogate,
    TwissParameterNetwork,
    InputNormalizer,
)


@pytest.fixture
def simple_simulator():
    """Create a simple simulator for testing."""
    segment = Segment(
        [
            Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
            Drift(name="D1", length=torch.tensor(1.0)),
            Screen(
                name="SCR1",
                resolution=(64, 64),
                pixel_size=torch.tensor([10e-4, 10e-4]),
                method="cloud-in-cell",
                is_active=True,
            ),
        ]
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        default_beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0),
            beta_y=torch.tensor(1.0),
            emittance_x=torch.tensor(1e-7),
            emittance_y=torch.tensor(1e-7),
            num_particles=1000,
            energy=torch.tensor(100e6),
            total_charge=torch.tensor(1e-12),
        )

    return CheetahSimulator(
        segment=segment,
        initial_beam_distribution=default_beam,
    )


@pytest.fixture
def simple_input_variables():
    """Create simple input variables (1 scalar) for testing with bounds."""
    return [
        CheetahWritableScalarVariable(
            name="Q1_k",
            element_name="Q1",
            element_attribute="k1",
            value_range=(-2.0, 2.0),
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


class TestNeuralNetworkBeamSurrogate:
    """Test the NeuralNetworkBeamSurrogate class."""

    def test_initialization_with_input_size(
        self, simple_simulator, simple_input_variables
    ):
        """Test initialization with input_size."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )
        assert model is not None
        assert isinstance(model.network, TwissParameterNetwork)

    def test_initialization_with_network(
        self, simple_simulator, simple_input_variables
    ):
        """Test initialization with pre-instantiated network."""
        network = TwissParameterNetwork(input_size=1)
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            network=network,
        )
        assert model.network is network

    def test_initialization_requires_input_size(
        self, simple_simulator, simple_input_variables
    ):
        """Test that initialization fails without input_size or network."""
        with pytest.raises(ValueError):
            NeuralNetworkBeamSurrogate(
                simulator=simple_simulator,
                action_variables=simple_input_variables,
            )

    def test_get_network(self, simple_simulator, simple_input_variables):
        """Test getting the network."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )
        network = model.get_network()
        assert isinstance(network, TwissParameterNetwork)

    def test_twiss_params_to_beam(self, simple_simulator, simple_input_variables):
        """Test conversion from Twiss parameters to beam."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
            num_particles=1000,
        )

        twiss_params = torch.tensor([[1.0, 0.0, 1e-7, 1.0, 0.0, 1e-7, 100e6]])

        beam = model._twiss_params_to_beam(twiss_params)

        assert isinstance(beam, ParticleBeam)
        assert beam.num_particles == 1000

    def test_twiss_params_to_beam_handles_negative_values(
        self, simple_simulator, simple_input_variables
    ):
        """Test that negative values are handled correctly."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        twiss_params = torch.tensor([[-1.0, -0.5, -1e-7, -1.0, -0.5, -1e-7, -100e6]])

        beam = model._twiss_params_to_beam(twiss_params)

        assert isinstance(beam, ParticleBeam)
        assert beam.energy >= 1e6

    def test_cheetah_initial_particles_property(
        self, simple_simulator, simple_input_variables
    ):
        """Test the cheetah_initial_particles property."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        beam = model.cheetah_initial_particles

        assert isinstance(beam, ParticleBeam)
        assert beam.num_particles == 10000

    def test_cheetah_initial_particles_setter(
        self, simple_simulator, simple_input_variables
    ):
        """Test setting cheetah_initial_particles."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        test_beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(2.0),
            beta_y=torch.tensor(2.0),
            emittance_x=torch.tensor(1e-6),
            emittance_y=torch.tensor(1e-6),
            num_particles=5000,
            energy=torch.tensor(50e6),
        )

        with pytest.raises(ValueError, match="generated by the network"):
            model.cheetah_initial_particles = test_beam

    def test_get_input_vector(self, simple_simulator, simple_input_variables):
        """Test getting input vector from action variables."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        input_vec = model._get_input_vector()

        assert isinstance(input_vec, torch.Tensor)
        assert input_vec.numel() > 0

    def test_custom_parameters(self, simple_simulator, simple_input_variables):
        """Test custom beam parameters."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
            num_particles=5000,
            total_charge=torch.tensor(2e-12),
        )

        assert model.num_particles == 5000
        assert model.total_charge is not None

        beam = model.cheetah_initial_particles
        assert beam.num_particles == 5000

    def test_device_and_dtype(self, simple_simulator, simple_input_variables):
        """Test device and dtype settings."""
        device = torch.device("cpu")
        dtype = torch.float32

        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
            device=device,
            dtype=dtype,
        )

        assert model.device == device
        assert model.dtype == dtype

    def test_supported_variables(self, simple_simulator, simple_input_variables):
        """Test supported_variables property."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        supported = model.supported_variables
        assert len(supported) == 1
        assert "Q1_k" in supported


class TestGradientFlow:
    """Test gradient flow through the model."""

    def test_gradients_propagate_to_beam(
        self, simple_simulator, simple_input_variables
    ):
        """Test that gradients propagate from beam coordinates to network parameters."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        # Get beam (which uses network)
        beam = model.cheetah_initial_particles

        # Compute a loss on beam properties
        loss = beam.mu_x.sum()

        # Backward should work
        loss.backward()

        # Check that network parameters have gradients
        for param in model.network.parameters():
            assert param.grad is not None

    def test_input_affects_beam_properties(
        self, simple_simulator, simple_input_variables
    ):
        """Test that changing network weights affects beam properties."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        # Get beam with default input
        beam1 = model.cheetah_initial_particles
        sigma_x1 = beam1.sigma_x.clone()

        # Manually change network weights significantly
        for param in model.network.parameters():
            param.data *= 2.0

        # Get beam with modified network
        beam2 = model.cheetah_initial_particles
        sigma_x2 = beam2.sigma_x.clone()

        # Beams should be different (sigma should change)
        assert not torch.allclose(sigma_x1, sigma_x2, rtol=0.1)

    def test_multiple_beams_independent(self, simple_simulator, simple_input_variables):
        """Test that multiple beam generations are independent."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        beam1 = model.cheetah_initial_particles
        beam2 = model.cheetah_initial_particles

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
        var = CheetahWritableScalarVariable(
            name="Q1_k",
            element_name="Q1",
            element_attribute="k1",
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
        var1 = CheetahWritableScalarVariable(
            name="var1",
            element_name="elem",
            element_attribute="attr",
            value_range=(0.0, 10.0),
        )
        var2 = CheetahWritableScalarVariable(
            name="var2",
            element_name="elem",
            element_attribute="attr",
            value_range=(-5.0, 5.0),
        )

        normalizer = InputNormalizer([var1, var2])

        # Test center points
        values = torch.tensor([[5.0, 0.0]])
        normalized = normalizer.normalize(values)
        expected = torch.tensor([[0.0, 0.0]])
        assert torch.allclose(normalized, expected)

    def test_normalized_input_in_model(self, simple_simulator, simple_input_variables):
        """Test that model uses normalized inputs."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        input_vec = model._get_input_vector()

        # Normalized input should be in [-1, 1]
        assert torch.all(input_vec >= -1.0) and torch.all(input_vec <= 1.0)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_public_batched_inputs_preserve_beams_and_gradients(
    simple_simulator, simple_input_variables, batch_size
):
    network = torch.nn.Linear(1, 7, dtype=torch.float64)
    with torch.no_grad():
        network.weight.zero_()
        network.weight[0, 0] = 0.2
        network.weight[2, 0] = 0.1
        network.bias.copy_(
            torch.tensor(
                [2.0, 0.3, -18.0, 3.0, -0.2, -18.0, 100e6], dtype=torch.float64
            )
        )
    model = NeuralNetworkBeamSurrogate(
        simple_simulator,
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
        loss = model.cheetah_final_particles.sigma_x.square().sum()
        loss.backward()
        assert inputs.grad is not None and (inputs.grad != 0).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        model.zero_grad()
        inputs.grad = None
    model.reset()
    torch.testing.assert_close(
        model.get(["Q1_k"])["Q1_k"], torch.tensor(1.0, dtype=torch.float64)
    )
    model.to(dtype=torch.float32)
    assert model.cheetah_final_particles.particles.dtype == torch.float32


class TestLogSpaceEmittance:
    """Test log-space emittance handling."""

    def test_log_emittance_transformation(
        self, simple_simulator, simple_input_variables
    ):
        """Test that emittances are correctly exponentialized from log space."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        # Network outputs in log space for emittances
        # [beta_x, alpha_x, log(emit_x), beta_y, alpha_y, log(emit_y), energy]
        log_emit_x = torch.log(torch.tensor(1e-8))
        log_emit_y = torch.log(torch.tensor(1e-8))
        twiss_params = torch.tensor(
            [[1.0, 0.0, log_emit_x, 1.0, 0.0, log_emit_y, 100e6]]
        )

        beam = model._twiss_params_to_beam(twiss_params)

        # Check that emittances are correctly transformed
        assert beam.emittance_x > 0
        assert beam.emittance_y > 0
        assert torch.allclose(beam.emittance_x, torch.tensor(1e-8), rtol=1e-5)

    def test_negative_log_values_handled(
        self, simple_simulator, simple_input_variables
    ):
        """Test that negative log values (sub-unity emittances) work correctly."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        # Negative log values correspond to small emittances
        twiss_params = torch.tensor([[1.0, 0.0, -20.0, 1.0, 0.0, -20.0, 100e6]])

        beam = model._twiss_params_to_beam(twiss_params)

        # exp(-20) is very small but positive
        expected_emit = torch.exp(torch.tensor(-20.0))
        assert beam.emittance_x > 0
        assert torch.allclose(beam.emittance_x, expected_emit, rtol=1e-5)

    def test_gradient_flow_through_log_emittance(
        self, simple_simulator, simple_input_variables
    ):
        """Test that gradients flow through log-space emittance transformation."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        beam = model.cheetah_initial_particles

        # Compute loss on emittance
        loss = beam.emittance_x.sum()
        loss.backward()

        # Gradients should propagate to network parameters
        for param in model.network.parameters():
            assert param.grad is not None, (
                "Gradients did not propagate through log-space transformation"
            )

    def test_emittance_values_reasonable(
        self, simple_simulator, simple_input_variables
    ):
        """Test that emittance values are always positive (exponential of real numbers)."""
        model = NeuralNetworkBeamSurrogate(
            simulator=simple_simulator,
            action_variables=simple_input_variables,
            input_size=1,
        )

        # Generate multiple beams with random network initialization
        for _ in range(5):
            beam = model.cheetah_initial_particles
            # Emittances should always be positive (exponential of any real number)
            assert beam.emittance_x > 0
            assert beam.emittance_y > 0
