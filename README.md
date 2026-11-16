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

`TorchAffineTransform` subclasses both `AffineTransform` and `torch.nn.Module`.
Both transforms compute `scale * value + offset` in `transform()` and
`(value - offset) / scale` in `untransform()`, initialized to identity.
Its PyTorch `forward()` delegates to `transform()`, preserving module calls.
The Torch subclass allows positive or negative scales and can change sign between
optimizer steps. Both transforms reject zero scales and nonfinite coefficients
at construction and evaluation; if an optimizer step lands exactly on zero,
evaluation raises rather than clipping it. `scale` and `offset` are the
independent scalar parameters (no cross-variable coupling), registered under
`wrapped.transforms`, not on the underlying model. Initialization matches the first floating-point
model parameter/buffer's dtype and device. Custom transform modules must
implement both `transform()` and `untransform()`.
The `transform_parameters` property returns a fresh list of unique, trainable
transform parameters, excluding other simulator parameters and frozen transforms.
Simulator parameters can be collected from `fitting_model` before or after
wrapping without including transform parameters. Alternatively, use
`get_trainable_parameters(wrapped)` for a single combined group.
Pass transforms explicitly to the constructor; the Torch mapping must be nonempty.
The generic wrapper owns a copy of the mapping, and the Torch wrapper owns a
`ModuleDict`. Neither reads or replaces an underlying model's `transforms` attribute.
The standalone `add_parameter_transforms` helper has been removed; use a mapping
comprehension in the constructor for factory-style construction.
For scale-only fitting, freeze each transform's offset with
`transform.offset.requires_grad_(False)` before creating the optimizer.
For offset-only fitting, freeze `transform.scale` instead. Both coefficients
are directly learnable parameters. State dicts now store `scale` rather than
`raw_scale`; to migrate an older positive-scale checkpoint, convert each
`raw_scale` value with `softplus(raw_scale) + torch.finfo(raw_scale.dtype).tiny`
and rename its key to `scale`.

- Transforms run at the **LUME public interface**, not inside Cheetah. Attaching a
  module alone does not execute it: always call the wrapper's `set()`.
- `set()` applies `transform()` to configured values; `get()` applies `untransform()`
  to configured readbacks. Unconfigured values pass through unchanged.
  The base model validates native values and retains its normal errors.
  The wrapper calls these methods directly, not a module's mode-dependent
  `forward()` dispatcher, so mappings apply consistently during training and evaluation.
- Read-only outputs can also be configured (for example, include `screen_image`
  in the constructor's mapping). Their `get()` values are inverse-transformed;
  attempting to set them still raises the model's read-only error.
- Training inputs **and targets** must be expressed in the wrapper's external
  coordinates. Other model APIs remain available through `wrapped.model`.
  The wrapper exposes the base model's native-coordinate
  `supported_variables` and inherits LUME's `set_value()`/`get_value()` helpers.
  `reset()` resets the underlying simulator without resetting fitted transforms.
  Simulator-specific subclass APIs are not forwarded automatically.
- This changes readback behavior: previously `get()` returned native values
  unchanged and plain callables were accepted. Replace those callables with
  bidirectional transform instances; use `wrapped.model.get()` for native values.
- The Torch wrapper is an `nn.Module`; normal `parameters()`, `state_dict()`,
  `train()`/`eval()`, and `to()` recurse through the base model and transforms.
  Recreate the wrapper with the same transform structure before loading a state
  dict. Transform keys now start with `transforms.` rather than
  `model.transforms.`; older checkpoints need those keys renamed.
  There is no separate serialization format.
- Attach transforms and move the model before constructing the optimizer.
  Later additions require updating/rebuilding the optimizer. Call `set()` with
  fresh inputs after an optimizer step to recompute the graph.
- Autodiff requires a differentiable simulator and transforms: the wrapper cannot
  repair a detached tensor or a NumPy-only simulation. Joint fitting also needs
  sufficiently varied scans; low loss alone does not prove unique identification
  of all scales, offsets, and misalignments. Different physical units may need
  separate optimizer learning rates.
- The currently tested Cheetah implementation produces nonfinite strength
  gradients at exactly `k1=0` (complex square-root backward). The example uses
  nonzero strengths. Transforms do not clip or perturb inputs to hide this;
  calibration raises on nonfinite gradients.

See [the parameter-transform notebook](docs/examples/cheetah_model_calibration_with_parameter_transforms.ipynb)
for joint transform and misalignment calibration, a misalignment-only
baseline, and inspection of the fitted parameters.

## Tests

After installing the `test` extra, run from the repository root:

```bash
python -m pytest lume_applications/tests -q
```

For CPU-only runs on systems with incompatible CUDA drivers, set
`CUDA_VISIBLE_DEVICES=''` before starting Python. In a notebook, set it before
importing PyTorch and restart the kernel if CUDA has already been initialized.
