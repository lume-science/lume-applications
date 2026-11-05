from copy import deepcopy

import pytest
import torch
from lightning.pytorch import Trainer
from cheetah.accelerator import Drift, Quadrupole, Screen, Segment
from cheetah.particles import ParticleBeam
from lume_cheetah.actions import (
	CheetahReadOnlyNDVariable,
	CheetahWritableScalarVariable,
)
from lume_cheetah.model import LUMECheetahModel
from lume_cheetah.simulator import CheetahSimulator

from lume_applications.model_calibration import (
	CalibrationModule,
	calibrate_model,
	get_trainable_parameters,
)


@pytest.fixture
def model() -> LUMECheetahModel:
	segment = Segment(
		[
			Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
			Drift(name="D1", length=torch.tensor(1.0)),
			Screen(
				name="SCR1",
				resolution=(64, 64),
				pixel_size=torch.tensor([1e-4, 1e-4]),
				method="cloud-in-cell",
				is_active=True,
			),
		]
	)

	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(0)
		particle_beam = ParticleBeam.from_twiss(
			beta_x=torch.tensor(1.0),
			beta_y=torch.tensor(1.0),
			emittance_x=torch.tensor(1e-7),
			emittance_y=torch.tensor(1e-7),
			num_particles=1000,
			energy=torch.tensor(100e6),
			total_charge=torch.tensor(1e-12),
		)

	simulator = CheetahSimulator(
		segment=segment, initial_beam_distribution=particle_beam
	)
	variables = [
		CheetahWritableScalarVariable(
			name="Q1_k", element_name="Q1", element_attribute="k1"
		),
		CheetahReadOnlyNDVariable(
			name="screen_image",
			element_name="SCR1",
			element_attribute="reading",
			shape=(64, 64),
		),
	]
	return LUMECheetahModel(simulator, action_variables=variables)


@pytest.fixture
def quadrupole_scan_dataset(model) -> dict[str, torch.Tensor]:
	"""Screen images versus k1 (m^-2), with a known transverse offset in meters."""
	offset = torch.tensor([2e-3, -3e-3])
	pixel_size = torch.tensor([2e-4, 2e-4])
	model.simulator.segment.Q1.misalignment = offset.clone()
	model.simulator.segment.SCR1.pixel_size = pixel_size.clone()
	strengths = torch.linspace(-2.0, 2.0, 9)
	images = []
	for strength in strengths:
		model.set({"Q1_k": strength})
		images.append(model.get(["screen_image"])["screen_image"].detach().clone())

	return {
		"quadrupole_k1": strengths,
		"quadrupole_offset_m": offset,
		"pixel_size_m": pixel_size,
		"images": torch.stack(images),
	}


def test_quadrupole_scan_dataset(quadrupole_scan_dataset, model):
	dataset = quadrupole_scan_dataset
	images = dataset["images"]
	assert images.shape == (9, 64, 64)
	torch.testing.assert_close(dataset["quadrupole_k1"], torch.linspace(-2.0, 2.0, 9))
	torch.testing.assert_close(
		dataset["quadrupole_offset_m"], torch.tensor([2e-3, -3e-3])
	)
	assert torch.isfinite(images).all()
	assert (images >= 0).all()
	assert (images.sum(dim=(-2, -1)) > 0).all()
	assert ((images[1:] - images[:-1]).abs().sum(dim=(-2, -1)) > 0).all()

	model.simulator.segment.Q1.misalignment = torch.zeros(2)
	model.set({"Q1_k": dataset["quadrupole_k1"][-1]})
	aligned_image = model.get(["screen_image"])["screen_image"]
	assert not torch.equal(images[-1], aligned_image)


def test_get_trainable_parameters(model):
	assert get_trainable_parameters(model) == []
	quadrupole = model.simulator.segment.Q1
	screen = model.simulator.segment.SCR1
	quadrupole.misalignment = torch.nn.Parameter(torch.zeros(2))
	screen.misalignment = torch.nn.Parameter(torch.zeros(2))
	screen.pixel_size = torch.nn.Parameter(screen.pixel_size, requires_grad=False)
	quadrupole.k1.requires_grad_(True)
	model.shared_offset = quadrupole.misalignment

	parameters = get_trainable_parameters(model)
	assert len(parameters) == 2
	assert {id(parameter) for parameter in parameters} == {
		id(quadrupole.misalignment), id(screen.misalignment),
	}
	quadrupole.misalignment.requires_grad_(False)
	screen.misalignment.requires_grad_(False)
	assert get_trainable_parameters(model) == []


def test_calibration_single_step(quadrupole_scan_dataset, model):
	dataset = quadrupole_scan_dataset
	original_data = {name: value.clone() for name, value in dataset.items()}
	original_state = {name: value.clone() for name, value in model.state_dict().items()}
	fitting_model = deepcopy(model)
	assert fitting_model.simulator.segment is not model.simulator.segment
	source_beam = model.simulator.beam_distribution.particles
	fitting_beam = fitting_model.simulator.beam_distribution.particles
	assert source_beam.data_ptr() != fitting_beam.data_ptr()
	initial_beam = fitting_beam.clone()
	fixed_geometry = [
		fitting_model.simulator.segment.Q1.length,
		fitting_model.simulator.segment.D1.length,
		fitting_model.simulator.segment.SCR1.pixel_size,
		fitting_model.simulator.segment.SCR1.misalignment,
	]
	original_geometry = [value.clone() for value in fixed_geometry]
	offset = torch.nn.Parameter(torch.zeros_like(fitting_model.simulator.segment.Q1.misalignment))
	fitting_model.simulator.segment.Q1.misalignment = offset
	assert dict(fitting_model.simulator.segment.Q1.named_parameters())["misalignment"] is offset
	assert any(parameter is offset for parameter in fitting_model.parameters())
	assert offset.data_ptr() != model.simulator.segment.Q1.misalignment.data_ptr()
	optimizer = torch.optim.Adam(get_trainable_parameters(fitting_model), lr=1e-4)
	scale = dataset["images"].abs().max()
	assert torch.isfinite(scale) and scale > 0

	def image_loss(prediction, target):
		return torch.nn.functional.mse_loss(prediction / scale, target / scale)

	history = calibrate_model(
		fitting_model,
		{"Q1_k": dataset["quadrupole_k1"]},
		{"screen_image": dataset["images"]},
		optimizer,
		steps=1,
		loss_fn=image_loss,
	)
	assert optimizer.param_groups[0]["params"][0] is offset
	assert fitting_model.simulator.segment.Q1.misalignment is offset
	assert len(history) == 2
	assert all(isinstance(loss, float) for loss in history)
	assert torch.isfinite(torch.tensor(history)).all()
	assert offset.grad is not None
	assert torch.isfinite(offset.grad).all()
	assert (offset.grad != 0).all()
	assert (offset.detach() != 0).all()
	with torch.no_grad():
		final_losses = []
		for strength, target in zip(dataset["quadrupole_k1"], dataset["images"]):
			fitting_model.set({"Q1_k": strength})
			final_losses.append(image_loss(fitting_model.get(["screen_image"])["screen_image"], target))
	assert history[-1] == pytest.approx(torch.stack(final_losses).mean().item())
	for name, value in model.state_dict().items():
		torch.testing.assert_close(value, original_state[name], equal_nan=True)
	for name, value in dataset.items():
		torch.testing.assert_close(value, original_data[name])
	torch.testing.assert_close(fitting_beam, initial_beam)
	for value, original in zip(fixed_geometry, original_geometry):
		torch.testing.assert_close(value, original)


def test_svi_misalignment(quadrupole_scan_dataset, model):
	import pyro
	import pyro.distributions as dist
	from pyro.nn import PyroSample
	from lume_applications.model_calibration.svi import (
		BayesianModel, infer_model_svi, predict_model_svi, _ObservationModel,
	)

	dataset = quadrupole_scan_dataset
	fitting_model = BayesianModel(
		deepcopy(model), {"misalignment": "simulator.segment.Q1.misalignment"},
	)
	fitting_model.misalignment = PyroSample(
		dist.Normal(torch.zeros(2), torch.full((2,), 3e-3)).to_event(1)
	)
	inputs = {"Q1_k": dataset["quadrupole_k1"]}
	targets = {"screen_image": dataset["images"]}
	noise_scales = {"screen_image": 0.05 * dataset["images"].max()}
	with pyro.get_param_store().scope(), torch.random.fork_rng(devices=[]):
		torch.manual_seed(0)
		marker = pyro.param("unrelated", torch.tensor(1.))
		observation_model = _ObservationModel(fitting_model, inputs, targets, noise_scales)
		trace = pyro.poutine.trace(observation_model).get_trace()
		latents = [
			name for name, site in trace.nodes.items()
			if site["type"] == "sample" and not site["is_observed"]
		]
		assert len(latents) == 1
		assert len([site for site in trace.nodes.values() if site.get("is_observed")]) == 9
		torch.testing.assert_close(
			fitting_model.model.simulator.segment.Q1.misalignment,
			trace.nodes[latents[0]]["value"],
		)
		result = infer_model_svi(
			fitting_model, inputs, targets, noise_scales,
			steps=2, num_samples=8, init_scale=1e-4,
			optimizer=pyro.optim.Adam({"lr": 1e-4}),
		)
		images = predict_model_svi(
			fitting_model, inputs, ["screen_image"], result.posterior_samples, num_samples=2,
		)["screen_image"]
		assert images.shape == (2, 9, 64, 64)
		assert not images.requires_grad
		assert torch.isfinite(images).all()
		known_images = predict_model_svi(
			fitting_model, inputs, ["screen_image"],
			{"model.misalignment": torch.stack([
				dataset["quadrupole_offset_m"], torch.zeros(2),
			])},
		)["screen_image"]
		torch.testing.assert_close(known_images[0], dataset["images"], atol=0, rtol=1e-5)
		assert not torch.equal(known_images[0], known_images[1])
		assert list(pyro.get_param_store().keys()) == ["unrelated"]
		assert pyro.param("unrelated") is marker
	assert len(result.losses) == 2
	assert torch.isfinite(torch.tensor(result.losses)).all()
	assert len(result.posterior_samples) == 1
	samples = next(iter(result.posterior_samples.values()))
	assert samples.shape == (8, 2)
	assert torch.isfinite(samples).all()
	assert (samples.std(dim=0) > 0).all()
	torch.testing.assert_close(model.simulator.segment.Q1.misalignment, dataset["quadrupole_offset_m"])


@pytest.mark.parametrize(
	"inputs, targets, steps, message",
	[
		({}, {"screen_image": torch.zeros(1, 64, 64)}, 1, "nonempty"),
		({"Q1_k": torch.ones(1)}, {}, 1, "nonempty"),
		({"Q1_k": torch.ones(2)}, {"screen_image": torch.zeros(1, 64, 64)}, 1, "sample counts"),
		({"Q1_k": torch.ones(0)}, {"screen_image": torch.zeros(0, 64, 64)}, 1, "sample counts"),
		({"Q1_k": torch.tensor(1.)}, {"screen_image": torch.zeros(1, 64, 64)}, 1, "sample dimension"),
		({"Q1_k": torch.ones(1)}, {"screen_image": torch.zeros(1, 64, 64)}, 0, "positive integer"),
		({"Q1_k": torch.ones(1)}, {"screen_image": torch.zeros(1, 1, 64)}, 1, "target shape"),
	],
)
def test_calibration_invalid_data(model, inputs, targets, steps, message):
	offset = torch.nn.Parameter(torch.zeros(2))
	model.simulator.segment.Q1.misalignment = offset
	optimizer = torch.optim.SGD([offset], lr=1e-4)
	with pytest.raises(ValueError, match=message):
		calibrate_model(model, inputs, targets, optimizer, steps=steps)
	torch.testing.assert_close(offset.detach(), torch.zeros(2))


@pytest.fixture
def additive_model():
	class AdditiveModel(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.offset = torch.nn.Parameter(torch.tensor(0.))
			self.gain = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)

		def set(self, inputs):
			self.prediction = self.gain * inputs["first"] + inputs["second"] + self.offset

		def get(self, names):
			outputs = {"value": self.prediction, "double": 2 * self.prediction}
			return {name: outputs[name] for name in names}

	return AdditiveModel()


@pytest.mark.parametrize("use_trainer", [False, True])
def test_calibration_default_loss(additive_model, use_trainer):
	inputs = {
		"first": torch.tensor([1., 2.], requires_grad=True),
		"second": torch.tensor([3., 4.], requires_grad=True),
	}
	targets = {
		"value": torch.tensor([6., 8.], requires_grad=True),
		"double": torch.tensor([12., 16.], requires_grad=True),
	}
	optimizer = torch.optim.SGD([additive_model.offset], lr=0.1)
	if use_trainer:
		calibration = CalibrationModule(additive_model, inputs, targets, optimizer)
		trainer = Trainer(
			accelerator="cpu", devices=1, max_steps=2, max_epochs=-1,
			logger=False, enable_checkpointing=False, enable_progress_bar=False,
			enable_model_summary=False,
		)
		trainer.fit(calibration)
		history = calibration.loss_history
		assert calibration.automatic_optimization
		assert trainer.global_step == 2
		assert "train_loss" in trainer.callback_metrics
	else:
		history = calibrate_model(additive_model, inputs, targets, optimizer, steps=2)
	assert history == pytest.approx([10., 2.5, 0.625])
	assert additive_model.offset.item() == pytest.approx(1.5)
	assert additive_model.gain.item() == 1.
	assert additive_model.gain.grad is None
	assert all(value.grad is None for value in [*inputs.values(), *targets.values()])


@pytest.fixture
def bayesian_additive_model(additive_model):
	import pyro.distributions as dist
	from pyro.nn import PyroSample
	from lume_applications.model_calibration.svi import BayesianModel

	model = BayesianModel(additive_model, {"offset": "offset"})
	model.offset = PyroSample(dist.Normal(torch.tensor(0.), torch.tensor(1.)))
	return model


def test_svi_predictions(bayesian_additive_model):
	from lume_applications.model_calibration.svi import predict_model_svi

	inputs = {
		"first": torch.tensor([1., 2.], requires_grad=True),
		"second": torch.tensor([3., 4.], requires_grad=True),
	}
	draws = {"model.offset": torch.tensor([0., 1., 2.], requires_grad=True)}
	prior = bayesian_additive_model._pyro_samples["offset"]
	prior_mean = prior.mean.clone()
	predictions = predict_model_svi(
		bayesian_additive_model, inputs, ["value", "double"], draws, num_samples=2,
	)
	expected = torch.tensor([[4., 6.], [5., 7.]])
	torch.testing.assert_close(predictions["value"], expected)
	torch.testing.assert_close(predictions["double"], 2 * expected)
	assert all(not value.requires_grad for value in predictions.values())
	torch.testing.assert_close(draws["model.offset"].detach(), torch.tensor([0., 1., 2.]))
	torch.testing.assert_close(inputs["first"].detach(), torch.tensor([1., 2.]))
	torch.testing.assert_close(inputs["second"].detach(), torch.tensor([3., 4.]))
	torch.testing.assert_close(prior.mean, prior_mean)
	assert all(value.grad is None for value in [*inputs.values(), *draws.values()])


@pytest.mark.parametrize(
	"samples, num_samples, message",
	[
		({}, None, "latent sites"),
		({"model.wrong": torch.zeros(2)}, None, "latent sites"),
		({"model.offset": torch.zeros(0)}, None, "draw counts"),
		({"model.offset": torch.tensor(0.)}, None, "draw dimension"),
		({"model.offset": torch.zeros(2)}, 3, "num_samples"),
		({"model.offset": torch.zeros(2)}, 0, "num_samples"),
	],
)
def test_svi_prediction_invalid_samples(bayesian_additive_model, samples, num_samples, message):
	from lume_applications.model_calibration.svi import predict_model_svi

	with pytest.raises(ValueError, match=message):
		predict_model_svi(
			bayesian_additive_model,
			{"first": torch.zeros(1), "second": torch.zeros(1)},
			["value"], samples, num_samples=num_samples,
		)


def test_svi_gaussian_posterior(bayesian_additive_model):
	import pyro
	from lume_applications.model_calibration.svi import infer_model_svi

	prior = bayesian_additive_model._pyro_samples["offset"]
	prior_mean = prior.mean.clone()
	prior_scale = prior.stddev.clone()
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(0)
		result = infer_model_svi(
			bayesian_additive_model,
			{"first": torch.zeros(2), "second": torch.zeros(2)},
			{"value": torch.full((2,), 2.)},
			{"value": 0.5},
			steps=2000, num_samples=2000, init_scale=0.2,
			optimizer=pyro.optim.Adam({"lr": 0.01}),
		)
	samples = result.posterior_samples["model.offset"]
	assert samples.shape == (2000,)
	assert not samples.requires_grad
	assert samples.mean().item() == pytest.approx(16 / 9, abs=0.15)
	assert samples.std().item() == pytest.approx(1 / 3, abs=0.1)
	assert sum(result.losses[-50:]) < sum(result.losses[:50])
	assert "offset" not in dict(bayesian_additive_model.model.named_parameters())
	torch.testing.assert_close(prior.mean, prior_mean)
	torch.testing.assert_close(prior.stddev, prior_scale)


@pytest.mark.parametrize("noise_scale", [0., -1., float("nan"), float("inf")])
def test_svi_invalid_noise(bayesian_additive_model, noise_scale):
	from lume_applications.model_calibration.svi import infer_model_svi

	with pytest.raises(ValueError, match="noise scales"):
		infer_model_svi(
			bayesian_additive_model,
			{"first": torch.zeros(1), "second": torch.zeros(1)},
			{"value": torch.zeros(1)}, {"value": noise_scale}, steps=1,
		)


def test_svi_requires_prior(additive_model):
	from lume_applications.model_calibration.svi import BayesianModel, infer_model_svi

	with pytest.raises(ValueError, match="PyroSample"):
		infer_model_svi(
			BayesianModel(additive_model, {"offset": "offset"}),
			{"first": torch.zeros(1), "second": torch.zeros(1)},
			{"value": torch.zeros(1)}, {"value": 1.}, steps=1,
		)


@pytest.mark.parametrize(
	"loss_fn, message",
	[
		(lambda prediction, target: prediction.square().reshape(1), "scalar tensor"),
		(lambda prediction, target: prediction * float("nan"), "loss must be finite"),
		(lambda prediction, target: prediction.detach(), "disconnected"),
		(lambda prediction, target: prediction.sqrt(), "gradients must be finite"),
	],
)
def test_calibration_invalid_loss(additive_model, loss_fn, message):
	optimizer = torch.optim.SGD([additive_model.offset], lr=0.1)
	with pytest.raises(ValueError, match=message):
		calibrate_model(
			additive_model,
			{"first": torch.zeros(1), "second": torch.zeros(1)},
			{"value": torch.zeros(1)},
			optimizer,
			steps=1,
			loss_fn=loss_fn,
		)
	assert additive_model.offset.item() == 0.


def test_calibration_disconnected_parameter(additive_model):
	unrelated = torch.nn.Parameter(torch.tensor(0.))
	optimizer = torch.optim.SGD([additive_model.offset, unrelated], lr=0.1)
	with pytest.raises(ValueError, match="optimizer parameter is disconnected"):
		calibrate_model(
			additive_model,
			{"first": torch.ones(1), "second": torch.zeros(1)},
			{"value": torch.zeros(1)},
			optimizer,
			steps=1,
		)
	assert additive_model.offset.item() == 0.


def test_model_tracks_to_screen(model):
	assert [element.name for element in model.simulator.segment.elements] == [
		"Q1",
		"D1",
		"SCR1",
	]

	model.set({"Q1_k": torch.tensor(2.0)})

	values = model.get(["Q1_k", "screen_image"])
	assert values["Q1_k"] == torch.tensor(2.0)
	image = values["screen_image"]
	assert image.shape == (64, 64)
	assert torch.isfinite(image).all()
	assert image.sum() > 0
