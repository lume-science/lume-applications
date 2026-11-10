from collections.abc import Callable, Mapping

import torch
from lightning.pytorch import LightningModule, Trainer
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from .utils import (
    clone_tensor_mapping,
    validate_loss_weights,
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
    """PyTorch Lightning training for a lume-style model
        (lume-torch, lume-cheetah) with a differentiable set(dict)/get(dict) interface.
        Trains any PyTorch parameters registered with the optimizer.

    Training inputs and targets are provided as mappings from lume-model variable
        names to tensors. By default each training batch contains the entire scan. Scalar
        losses are summed over samples for each target, then combined according to
        `loss_weights` (omitted weights default to 1). Reduction within each output
        is controlled by loss_fn (MSE by default).

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
        """
        Initialize the calibration module with the model, inputs, targets, optimizer,
        loss function, and optional loss weights.

        Parameters
        ----------
        model : torch.nn.Module
            The differentiable model to be calibrated.
        inputs : Mapping[str, torch.Tensor]
            The input tensors for the scan.
        targets : Mapping[str, torch.Tensor]
            The target tensors for the scan.
        optimizer : torch.optim.Optimizer
            The optimizer for updating model parameters.
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
            The loss function to use (default is mse_loss) for each target.
        loss_weights : Mapping[str, float] | None, optional
            Weights for each target in the loss computation (default is None).
        """

        super().__init__()

        validate_training_data(inputs, targets)
        self.loss_weights = validate_loss_weights(loss_weights, targets)

        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if not parameters or any(
            not parameter.requires_grad for parameter in parameters
        ):
            raise ValueError("optimizer must contain trainable parameters")
        self.model = model
        self.inputs = clone_tensor_mapping(inputs)
        self.targets = clone_tensor_mapping(targets)
        self._optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_history: list[float] = []
        self.target_loss_history: dict[str, list[float]] = {
            name: [] for name in targets
        }

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for the training data."""
        return DataLoader([(self.inputs, self.targets)], batch_size=None, num_workers=0)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the optimizer for training."""
        return self._optimizer

    def evaluate_loss(
        self, inputs, targets
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the total loss and differentiable weighted per-target scan sums.

        Parameters
        ----------
        inputs : Mapping[str, torch.Tensor]
                The input tensors for the scan.
        targets : Mapping[str, torch.Tensor]
                The target tensors for the scan.

        Returns
        -------
        total_loss : torch.Tensor
                The total weighted loss for the scan.
        target_losses : dict[str, torch.Tensor]
                The weighted per-target losses for the scan.

        """
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
        """
                Perform a single training step.

        Parameters
        ----------
        batch : tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
            A tuple containing the input and target tensors for the batch.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the total loss and per-target losses.

        """

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
            on_step=True,
            on_epoch=False,
        )
        return {"loss": loss, "target_losses": target_losses}

    def on_before_optimizer_step(self, optimizer):
        """
        Called before the optimizer step to validate parameter gradients.
        """
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    raise ValueError(
                        "an optimizer parameter is disconnected from the loss"
                    )
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError("parameter gradients must be finite")

    def on_train_start(self):
        """
        Called at the start of training to clear loss histories.
        """
        self.loss_history.clear()
        for history in self.target_loss_history.values():
            history.clear()

    def on_train_end(self):
        """
        Called at the end of training to perform a final evaluation and update loss histories.
        """
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
    """
        Fit in place a lume-style differentiable model with a single-CPU
        Lightning Trainer and full-scan updates.

        Returns steps + 1 weighted summed losses, including a fresh final evaluation. Use
        CalibrationModule with your own Trainer for per-target histories, device
        configuration, or callbacks.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be calibrated.
    inputs : Mapping[str, torch.Tensor]
        The input tensors for the calibration.
    targets : Mapping[str, torch.Tensor]
        The target tensors for the calibration.
    optimizer : torch.optim.Optimizer
        The optimizer used for the calibration.
    steps : int
        The number of optimization steps to perform.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        The loss function to use for calibration (default is mse_loss).
    loss_weights : Mapping[str, float] | None, optional
        The weights for each target loss (default is None).
    """
    validate_positive_integer(steps, "steps")
    calibration = CalibrationModule(
        model,
        inputs,
        targets,
        optimizer,
        loss_fn=loss_fn,
        loss_weights=loss_weights,
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
