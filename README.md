# LUME Applications

Reusable calibration and online control workflows for differentiable LUME simulation models. Fit
model parameters with PyTorch Lightning, infer parameter distributions with
Pyro stochastic variational inference (SVI), or use models to inform optimization in Xopt.

The included Cheetah example learns a quadrupole's transverse misalignment from
screen images collected over a quadrupole-strength scan. It uses an in-process
LUMEcheetah model; no EPICS server is required.

## Installation

Requires Python 3.10 or newer. From the repository root:

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

## Loss Tracking

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

## Tests

After installing the `test` extra, run from the repository root:

```bash
python -m pytest lume_applications/tests -q
```

For CPU-only runs on systems with incompatible CUDA drivers, set
`CUDA_VISIBLE_DEVICES=''` before starting Python. In a notebook, set it before
importing PyTorch and restart the kernel if CUDA has already been initialized.
