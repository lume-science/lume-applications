from collections.abc import Callable, Mapping

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
	losses are averaged equally over samples and outputs, with exact matching
	prediction/target shapes. Data are copied and detached.

	Register parameters before creating the supplied optimizer. Scanned inputs
	must not overwrite fitted parameters. The wrapped model is mutated in place
	and never reset. Lightning owns backward, gradient clearing and updates.
	Use a single-device Trainer with accumulate_grad_batches=1 for one update
	per full scan. loss_history contains one scalar per training batch followed
	by a fresh full-scan loss at the end of fit. train_loss is logged to Lightning.
	"""

	def __init__(
		self,
		model: torch.nn.Module,
		inputs: Mapping[str, torch.Tensor],
		targets: Mapping[str, torch.Tensor],
		optimizer: torch.optim.Optimizer,
		*,
		loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
	):
		super().__init__()
		validate_training_data(inputs, targets)
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

	def train_dataloader(self) -> DataLoader:
		return DataLoader([(self.inputs, self.targets)], batch_size=None, num_workers=0)

	def configure_optimizers(self) -> torch.optim.Optimizer:
		return self._optimizer

	def evaluate_loss(self, inputs, targets) -> torch.Tensor:
		losses = []
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
				losses.append(loss)
		return torch.stack(losses).mean()

	def training_step(self, batch, batch_idx):
		inputs, targets = batch
		loss = self.evaluate_loss(inputs, targets)
		if not loss.requires_grad:
			raise ValueError("loss is disconnected from the trainable parameters")
		self.loss_history.append(loss.detach().item())
		self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
		return loss

	def on_before_optimizer_step(self, optimizer):
		for group in optimizer.param_groups:
			for parameter in group["params"]:
				if parameter.grad is None:
					raise ValueError("an optimizer parameter is disconnected from the loss")
				if not torch.isfinite(parameter.grad).all():
					raise ValueError("parameter gradients must be finite")

	def on_train_start(self):
		self.loss_history.clear()

	def on_train_end(self):
		inputs, targets = self.transfer_batch_to_device(
			(self.inputs, self.targets), self.device, 0
		)
		with torch.no_grad():
			self.loss_history.append(self.evaluate_loss(inputs, targets).item())


def calibrate_model(
	model: torch.nn.Module,
	inputs: Mapping[str, torch.Tensor],
	targets: Mapping[str, torch.Tensor],
	optimizer: torch.optim.Optimizer,
	*,
	steps: int,
	loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
) -> list[float]:
	"""Fit in place with a single-CPU Lightning Trainer and full-scan updates.

	Returns steps + 1 losses, including a fresh final evaluation. Use
	CalibrationModule with your own Trainer for device or callback configuration.
	"""
	validate_positive_integer(steps, "steps")
	calibration = CalibrationModule(model, inputs, targets, optimizer, loss_fn=loss_fn)
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