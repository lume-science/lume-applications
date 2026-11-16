# LUME Applications

Reusable calibration and online control workflows for differentiable LUME simulation models. Fit
model parameters with PyTorch Lightning, infer parameter distributions with
Pyro stochastic variational inference (SVI), or use models to inform optimization in Xopt.

The included Cheetah example learns a quadrupole's transverse misalignment from
screen images collected over a quadrupole-strength scan. It uses an in-process
LUMEcheetah model; no EPICS server is required.

## Installation

Requires Python 3.11 or newer. From the repository root:

```bash
python -m pip install .
```

For an editable installation with tests and notebook dependencies:

```bash
python -m pip install -e '.[test,examples]'
```

Runtime dependencies include PyTorch, Lightning, Pyro, Cheetah, and LUMEcheetah.
The `test` extra adds pytest; `examples` adds matplotlib and ipykernel. Open the
example notebook in a notebook frontend such as VS Code and select the Python
environment where the package is installed.

## Model Calibration

`CalibrationModule` sums the scalar losses across scan points for each target,
then adds those target contributions to obtain the total. The supplied `loss_fn`
still controls reduction within each output; the default MSE averages image pixels.

- `evaluate_loss(inputs, targets)` returns `(total_loss, target_losses)`, with
	differentiable tensors keyed by target name in `target_losses`.
- `loss_history` records totals; `target_loss_history[name]` records each target's
	contribution at the same steps. Both include a fresh final evaluation.
- Lightning logs `train_loss` and `train_loss/<target>` during training.

`calibrate_model` continues to return only the total-loss history. Use
`CalibrationModule` directly to access per-target histories. Summing instead of
averaging increases loss and gradient scales with scan size and target count;
existing learning rates may need adjustment.

### Bidirectional calibration transforms

`ModelWithTransforms` inherits from lume-base's `LUMEModel` and wraps another
`LUMEModel`. Each configured transform must be an instance with both
`transform(value)` and `untransform(value)` methods (`BidirectionalTransform`),
matching BoTorch's `InputTransform` naming. `untransform` is the inverse mapping:

```python
from lume_applications.model_calibration import AffineTransform, ModelWithTransforms

wrapped = ModelWithTransforms(
    model, {"Q1_k": AffineTransform(scale=1.02, offset=0.01)}
)
wrapped.set({"Q1_k": 0.5})  # model receives 0.52
wrapped.get_value("Q1_k")  # returns approximately 0.5
```

`AffineTransform` uses ordinary scalar arithmetic without converting input values
to Torch. It works with scalar or array-based simulations, accepts finite positive
or negative scales (but not zero), and has no learnable parameters.
`get_transform_params()` reports its coefficients and `reset_to_identity()` resets them.

For differentiable models inheriting both `LUMEModel` and `torch.nn.Module`,
pass per-variable transform modules directly to `TorchModelWithTransforms`:

```python
from lume_applications.model_calibration import (
    TorchAffineTransform,
    TorchModelWithTransforms,
    get_trainable_parameters,
)

groups = [{"params": get_trainable_parameters(fitting_model), "lr": 1e-4}]
wrapped = TorchModelWithTransforms(
    fitting_model,
    transforms={name: TorchAffineTransform() for name in ("Q1_k", "Q2_k", "Q3_k")},
)
groups.append({"params": wrapped.transform_parameters, "lr": 1e-3})
optimizer = torch.optim.Adam(groups)
# Pass wrapped and optimizer to CalibrationModule as usual.
```


## Tests

After installing the `test` extra, run from the repository root:

```bash
python -m pytest lume_applications/tests -q
```

For CPU-only runs on systems with incompatible CUDA drivers, set
`CUDA_VISIBLE_DEVICES=''` before starting Python. In a notebook, set it before
importing PyTorch and restart the kernel if CUDA has already been initialized.
