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

## Tests

After installing the `test` extra, run from the repository root:

```bash
python -m pytest lume_applications/tests -q
```

For CPU-only runs on systems with incompatible CUDA drivers, set
`CUDA_VISIBLE_DEVICES=''` before starting Python. In a notebook, set it before
importing PyTorch and restart the kernel if CUDA has already been initialized.