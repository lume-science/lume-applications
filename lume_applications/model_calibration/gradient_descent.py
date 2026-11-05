from collections.abc import Callable, Mapping
import math
from numbers import Real

import torch
from lightning.pytorch import LightningModule, Trainer
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from .utils import (
	clone_tensor_mapping,
	validate_positive_integer,
	validate_prediction_shape,
	validate_training_data,
)


def get_trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
	"""Return registered parameters with requires_grad=True for an optimizer.

	Recurses through the model's submodules, excluding buffers and frozen
	parameters. Shared parameters appear once; objects are not copied. Returns
	an empty list if none are trainable. Call after registering parameters and
	before constructing the optimizer.
	"""
	return [parameter for parameter in model.parameters() if parameter.requires_grad]


class CalibrationModule(LightningModule):
	"""Lightning training for a differentiable set/get model and a fixed scan.

	Input and target keys are model variable names; tensors share a leading
	sample dimension. Each training batch contains the entire scan. Scalar
	losses are summed over samples for each target, then summed over targets,
	with exact matching prediction/target shapes. Reduction within each output
	is controlled by loss_fn (MSE by default). Data are copied and detached.
	loss_weights optionally maps target names to finite nonnegative multipliers.
	Omitted weights default to 1; at least one must be positive. Returned terms,
	histories, and logged target losses include these weights without normalization.

	Register parameters before creating the supplied optimizer. Scanned inputs
	must not overwrite fitted parameters. The wrapped model is mutated in place
	and never reset. Lightning owns backward, gradient clearing and updates.
	Use a single-device Trainer with accumulate_grad_batches=1 for one update
	per full scan. loss_history contains one scalar per training batch followed
	by a fresh full-scan loss at the end of fit. target_loss_history maps target
	names to aligned contribution histories. Lightning logs train_loss and
	train_loss/<target>. Increasing scan size or target count increases the loss
	and gradient scale; optimizer learning rates may need adjustment.
	"""

	def __init__(
		self,
		model: torch.nn.Module,
		inputs: Mapping[str, torch.Tensor],
		targets: Mapping[str, torch.Tensor],
		optimizer: torch.optim.Optimizer,
		*,
		loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
		loss_weights: Mapping[str, float] | None = None,
	):
		super().__init__()
		validate_training_data(inputs, targets)
		weights = {} if loss_weights is None else dict(loss_weights)
		if weights.keys() - targets.keys():
			raise ValueError("loss_weights contains unknown target names")
		if any(
			isinstance(weight, bool) or not isinstance(weight, Real)
			or not math.isfinite(weight) or weight < 0
			for weight in weights.values()
		):
			raise ValueError("loss_weights must be finite nonnegative numbers")
		self.loss_weights = {name: float(weights.get(name, 1.0)) for name in targets}
		if not any(self.loss_weights.values()):
			raise ValueError("at least one loss weight must be positive")
		parameters = [
			parameter for group in optimizer.param_groups for parameter in group["params"]
		]
		if not parameters or any(not parameter.requires_grad for parameter in parameters):
			raise ValueError("optimizer must contain trainable parameters")
		self.model = model
		self.inputs = clone_tensor_mapping(inputs)
		self.targets = clone_tensor_mapping(targets)
		self._optimizer = optimizer
		self.loss_fn = loss_fn
		self.loss_history: list[float] = []
		self.target_loss_history: dict[str, list[float]] = {name: [] for name in targets}

	def train_dataloader(self) -> DataLoader:
		return DataLoader([(self.inputs, self.targets)], batch_size=None, num_workers=0)

	def configure_optimizers(self) -> torch.optim.Optimizer:
		return self._optimizer

	def evaluate_loss(self, inputs, targets) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
		"""Return the total loss and differentiable weighted per-target scan sums."""
		losses = {name: [] for name in targets}
		for sample_index in range(next(iter(inputs.values())).shape[0]):
			self.model.set(
				{name: values[sample_index].clone() for name, values in inputs.items()}
			)
			predictions = self.model.get(list(targets))
			for name, values in targets.items():
				prediction = predictions[name]
				target = values[sample_index]
				validate_prediction_shape(prediction, target, name)
				loss = self.loss_fn(prediction, target)
				if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
					raise ValueError("loss_fn must return a scalar tensor")
				if not torch.isfinite(loss):
					raise ValueError("loss must be finite")
				losses[name].append(loss)
		target_losses = {
			name: self.loss_weights[name] * torch.stack(values).sum()
			for name, values in losses.items()
		}
		total_loss = torch.stack(list(target_losses.values())).sum()
		if not torch.isfinite(total_loss):
			raise ValueError("loss must be finite")
		return total_loss, target_losses

	def training_step(self, batch, batch_idx):
		inputs, targets = batch
		loss, target_losses = self.evaluate_loss(inputs, targets)
		if not loss.requires_grad:
			raise ValueError("loss is disconnected from the trainable parameters")
		self.loss_history.append(loss.detach().item())
		for name, target_loss in target_losses.items():
			self.target_loss_history[name].append(target_loss.detach().item())
		self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
		self.log_dict(
			{f"train_loss/{name}": value for name, value in target_losses.items()},
			on_step=True, on_epoch=False,
		)
		return {"loss": loss, "target_losses": target_losses}

	def on_before_optimizer_step(self, optimizer):
		for group in optimizer.param_groups:
			for parameter in group["params"]:
				if parameter.grad is None:
					raise ValueError("an optimizer parameter is disconnected from the loss")
				if not torch.isfinite(parameter.grad).all():
					raise ValueError("parameter gradients must be finite")

	def on_train_start(self):
		self.loss_history.clear()
		for history in self.target_loss_history.values():
			history.clear()

	def on_train_end(self):
		inputs, targets = self.transfer_batch_to_device(
			(self.inputs, self.targets), self.device, 0
		)
		with torch.no_grad():
			loss, target_losses = self.evaluate_loss(inputs, targets)
			self.loss_history.append(loss.item())
			for name, target_loss in target_losses.items():
				self.target_loss_history[name].append(target_loss.item())


def calibrate_model(
	model: torch.nn.Module,
	inputs: Mapping[str, torch.Tensor],
	targets: Mapping[str, torch.Tensor],
	optimizer: torch.optim.Optimizer,
	*,
	steps: int,
	loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
	loss_weights: Mapping[str, float] | None = None,
) -> list[float]:
	"""Fit in place with a single-CPU Lightning Trainer and full-scan updates.

	Returns steps + 1 weighted summed losses, including a fresh final evaluation. Use
	CalibrationModule with your own Trainer for per-target histories, device
	configuration, or callbacks.
	"""
	validate_positive_integer(steps, "steps")
	calibration = CalibrationModule(
		model, inputs, targets, optimizer, loss_fn=loss_fn, loss_weights=loss_weights,
	)
	trainer = Trainer(
		accelerator="cpu",
		devices=1,
		max_steps=steps,
		max_epochs=-1,
		logger=False,
		enable_checkpointing=False,
		enable_progress_bar=False,
		enable_model_summary=False,
	)
	trainer.fit(calibration)
	return calibration.loss_history