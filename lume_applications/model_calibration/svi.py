from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_mean
from pyro.nn import PyroModule, PyroParam, pyro_method
from pyro.optim import Adam, PyroOptim

from .utils import (
	clone_tensor_mapping,
	validate_finite_tensors,
	validate_positive_integer,
	validate_prediction_shape,
	validate_tensor_batch,
	validate_training_data,
)


@dataclass
class SVIResult:
	losses: list[float]
	posterior_samples: dict[str, torch.Tensor]


class BayesianModel(PyroModule):
	"""Bind Pyro attributes to tensor attributes of a LUME model clone.

	parameter_paths maps user-chosen names to dotted model attribute paths,
	e.g. {"misalignment": "simulator.segment.Q1.misalignment"}. Each exposed
	attribute starts as a PyroParam and can be replaced by PyroSample(prior).
	This adapter avoids converting LUME/Cheetah's abstract classes to PyroModule.
	"""

	def __init__(self, model: torch.nn.Module, parameter_paths: Mapping[str, str]):
		super().__init__()
		self.model = model
		self.parameter_paths = dict(parameter_paths)
		if not parameter_paths:
			raise ValueError("parameter_paths must be nonempty")
		for name, path in parameter_paths.items():
			if not name.isidentifier() or hasattr(self, name):
				raise ValueError(f"invalid or reserved parameter name: {name!r}")
			owner, attribute = self._resolve(path)
			value = getattr(owner, attribute)
			if not isinstance(value, torch.Tensor):
				raise TypeError(f"{path!r} must refer to a tensor attribute")
			setattr(self, name, PyroParam(value.detach().clone()))

	def _resolve(self, path):
		owner_path, _, attribute = path.rpartition(".")
		return self.model.get_submodule(owner_path), attribute

	@pyro_method
	def bind_parameters(self):
		for name, path in self.parameter_paths.items():
			owner, attribute = self._resolve(path)
			previous = getattr(owner, attribute)
			value = getattr(self, name)
			if value.shape != previous.shape:
				raise ValueError(f"sample for {name!r} must match shape {previous.shape}")
			if isinstance(previous, torch.nn.Parameter):
				delattr(owner, attribute)
				owner.register_buffer(attribute, value)
			else:
				setattr(owner, attribute, value)

	def set(self, inputs):
		self.model.set(inputs)

	def get(self, names):
		return self.model.get(names)


class _PredictionModel(PyroModule):
	def __init__(self, model, inputs, output_names):
		super().__init__()
		self.model = model
		self.inputs = inputs
		self.output_names = output_names

	def forward(self):
		self.model.bind_parameters()
		outputs = {name: [] for name in self.output_names}
		for sample_index in range(next(iter(self.inputs.values())).shape[0]):
			self.model.set({
				name: values[sample_index].clone() for name, values in self.inputs.items()
			})
			predictions = self.model.get(self.output_names)
			for name in self.output_names:
				value = predictions[name]
				if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
					raise ValueError(f"prediction for {name!r} must be a finite tensor")
				outputs[name].append(value.clone())
		for name, values in outputs.items():
			pyro.deterministic(f"predictions/{name}", torch.stack(values))


def predict_model_svi(
	model: BayesianModel,
	inputs: Mapping[str, torch.Tensor],
	output_names: Sequence[str],
	posterior_samples: Mapping[str, torch.Tensor],
	*,
	num_samples: int | None = None,
) -> dict[str, torch.Tensor]:
	"""Replay fitted SVI draws to predict outputs at arbitrary scan inputs.

	Pass result.posterior_samples from infer_model_svi with the same BayesianModel
	and latent names. Returns tensors shaped (draws, scan points, *output_shape).
	Each latent draw is shared across all scan points. Outputs are noise-free
	simulator predictions: average over dimension 0 for posterior-mean images,
	or take quantiles over it for uncertainty due to the inferred parameters.
	This differs from evaluating just the posterior-mean parameter value.

	Uses Pyro Predictive in sequential mode. No measurement noise is added and
	no ground-truth outputs are needed. num_samples selects the first draws to
	limit compute and memory; by default all supplied draws are evaluated.
	Inputs/draws are copied and detached, priors are preserved, and the caller's
	Pyro parameter store is untouched. The supplied simulation is evaluated in
	place and remains at the last draw/input; use the independent fitting clone.
	"""
	if not isinstance(model, BayesianModel):
		raise TypeError("model must be wrapped in BayesianModel before prediction")
	if not inputs or not output_names or isinstance(output_names, str):
		raise ValueError("inputs and output_names must be nonempty")
	if len(set(output_names)) != len(output_names):
		raise ValueError("output_names must be unique")
	values = list(inputs.values())
	validate_tensor_batch(values, name="input")
	expected_sites = {f"model.{name}" for name in model._pyro_samples}
	if not expected_sites or set(posterior_samples) != expected_sites:
		raise ValueError("posterior_samples must match all model latent sites")
	draws = list(posterior_samples.values())
	draw_count = validate_tensor_batch(draws, name="posterior", dimension="draw")
	validate_finite_tensors([*values, *draws], name="inputs and posterior samples")
	if num_samples is None:
		num_samples = draw_count
	if (
		not isinstance(num_samples, int) or isinstance(num_samples, bool)
		or not 0 < num_samples <= draw_count
	):
		raise ValueError("num_samples must be positive and no larger than the supplied draw count")
	posterior_samples = clone_tensor_mapping(posterior_samples, limit=num_samples)
	prediction_model = _PredictionModel(
		model, clone_tensor_mapping(inputs),
		list(output_names),
	)
	with pyro.get_param_store().scope(), torch.no_grad():
		predictions = Predictive(
			prediction_model, posterior_samples=posterior_samples,
			return_sites=[f"predictions/{name}" for name in output_names], parallel=False,
		)()
	return {name: predictions[f"predictions/{name}"].detach() for name in output_names}


class _ObservationModel(PyroModule):
	def __init__(self, model, inputs, targets, noise_scales):
		super().__init__()
		self.model = model
		self.inputs = inputs
		self.targets = targets
		self.noise_scales = noise_scales

	def forward(self):
		self.model.bind_parameters()
		for sample_index in range(next(iter(self.inputs.values())).shape[0]):
			self.model.set({
				name: values[sample_index].clone() for name, values in self.inputs.items()
			})
			predictions = self.model.get(list(self.targets))
			for name, values in self.targets.items():
				prediction = predictions[name]
				target = values[sample_index]
				validate_prediction_shape(prediction, target, name)
				pyro.sample(
					f"observations/{sample_index}/{name}",
					dist.Normal(prediction, self.noise_scales[name]).to_event(target.ndim),
					obs=target,
				)


def infer_model_svi(
	model: BayesianModel,
	inputs: Mapping[str, torch.Tensor],
	targets: Mapping[str, torch.Tensor],
	noise_scales: Mapping[str, float | torch.Tensor],
	*,
	steps: int,
	num_samples: int = 1000,
	optimizer: PyroOptim | None = None,
	init_scale: float = 0.1,
) -> SVIResult:
	"""Approximate a set/get model's posterior with full-scan Pyro SVI.

	Wrap an independent clone in BayesianModel and replace its exposed latent
	attributes with PyroSample(prior) before calling. A PyroParam alone is a
	point estimate, not a prior; at least one continuous latent is required.
	Each PyroSample draw is shared across the entire scan via a single enclosing
	PyroModule call. Use to_event for vector-valued priors such as x/y offsets.

	Inputs and targets have a matching leading sample dimension. noise_scales
	provides one positive scalar standard deviation per output, in target units.
	The likelihood assumes independent Gaussian errors across all samples and
	output components. It is summed, not averaged like an MSE training loss.

	AutoNormal fits a mean-field variational posterior, initialized at prior
	means with init_scale in latent units. The default optimizer is Pyro Adam
	with lr=0.01; supply a Pyro optimizer for other rates, especially for physical
	parameters measured in meters. Data are detached copies; the passed model
	is evaluated in place but its priors are not replaced by posterior means.

	Runs in a fresh parameter-store scope without clearing the caller's store.
	Returns detached posterior draws keyed by Pyro latent site name, with a
	leading num_samples dimension, and one negative-ELBO estimate per step.
	These draws describe the variational approximation, not an exact posterior.
	"""
	if not isinstance(model, BayesianModel):
		raise TypeError("model must be wrapped in BayesianModel before inference")
	for name, value in (("steps", steps), ("num_samples", num_samples)):
		validate_positive_integer(value, name)
	if not math.isfinite(init_scale) or init_scale <= 0:
		raise ValueError("init_scale must be finite and positive")
	validate_training_data(inputs, targets, require_finite=True)
	if noise_scales.keys() != targets.keys():
		raise ValueError("noise_scales must have exactly the target output names")
	scales = {}
	
	for name, target in targets.items():
		scale = torch.as_tensor(noise_scales[name], dtype=target.dtype, device=target.device)
		if scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
			raise ValueError("noise scales must be finite positive scalars")
		scales[name] = scale.detach().clone()
	observation_model = _ObservationModel(
		model,
		clone_tensor_mapping(inputs),
		clone_tensor_mapping(targets),
		scales,
	)
	with pyro.get_param_store().scope():
		guide = AutoNormal(
			observation_model,
			init_loc_fn=lambda site: init_to_mean(site).detach().clone(),
			init_scale=init_scale,
		)
		if not guide():
			raise ValueError("model must contain at least one PyroSample latent prior")
		svi = SVI(
			observation_model, guide,
			Adam({"lr": 0.01}) if optimizer is None else optimizer,
			loss=Trace_ELBO(),
		)
		losses = []
		for _ in range(steps):
			loss = svi.step()
			if not math.isfinite(loss):
				raise ValueError("SVI loss must be finite")
			losses.append(loss)
		with torch.no_grad():
			draws = [guide() for _ in range(num_samples)]
		posterior_samples = {
			name: torch.stack([draw[name].detach().clone() for draw in draws])
			for name in draws[0]
		}
	return SVIResult(losses, posterior_samples)