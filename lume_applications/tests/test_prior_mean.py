"""Check generalized_taylor and GenerizalizedTaylorPrior against hand-derived derivatives.

Each case pairs a toy model y = F(x) with a scalar g(y) whose gradient and Hessian of
f(x) = g(F(x)) are known in closed form, so the autograd result is compared to an analytic one.
"""

import math

import pytest
import torch
from lume.model import LUMEModel
from lume_torch.variables import TorchScalarVariable

from lume_applications.prior_mean import GenerizalizedTaylorPrior, generalized_taylor

DTYPE = torch.float64

A = torch.tensor([[1.0, -2.0, 0.5], [0.3, 1.5, -1.0]], dtype=DTYPE)  # (m=2, n=3)
B = torch.tensor([0.2, -0.4], dtype=DTYPE)
X0 = torch.tensor([0.3, -0.7, 1.1], dtype=DTYPE)


class AnalyticModel(LUMEModel):
	"""LUMEModel whose outputs are y = fn(x), recomputed on every set."""

	def __init__(self, fn, x0, n_outputs):
		self.fn = fn
		self.input_names = [f"x{i}" for i in range(len(x0))]
		self.output_names = [f"y{i}" for i in range(n_outputs)]
		self._variables = {
			name: TorchScalarVariable(name=name)
			for name in self.input_names
		} | {
			name: TorchScalarVariable(name=name, read_only=True)
			for name in self.output_names
		}
		self._x0 = x0.clone()
		self.reset()


	@property
	def supported_variables(self):
		return self._variables

	def _get(self, names):
		return {name: self._state[name] for name in names}

	def _set(self, values):
		x = torch.stack([values.get(k, self._state[k]) for k in self.input_names])
		y = self.fn(x)
		self._state = dict(zip(self.input_names, x.unbind())) | dict(zip(self.output_names, y.unbind()))

	def reset(self):
		self._state = dict(zip(self.input_names, self._x0.unbind()))
		self._set({})


# Each case: (model fn F, g, analytic grad of f at x, analytic Hessian of f at x)
def _linear(x):
	return A @ x + B

def _softmax(y):
	return torch.softmax(y, dim=-1)

CASES = {
	# f = |Ax + b|^2  ->  grad = 2 A^T y,  H = 2 A^T A  (exactly quadratic)
	"sum_of_squares": (
		_linear,
		lambda y: (y**2).sum(),
		lambda x: 2 * A.T @ _linear(x),
		lambda x: 2 * A.T @ A,
	),
	# f = y0 * y1 with y = Ax + b  ->  grad = A^T [y1, y0],  H = A^T [[0,1],[1,0]] A
	"cross_term": (
		_linear,
		lambda y: y[0] * y[1],
		lambda x: A.T @ _linear(x).flip(0),
		lambda x: A.T @ torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE) @ A,
	),
	# f = logsumexp(Ax + b)  ->  grad = A^T p,  H = A^T (diag(p) - p p^T) A,  p = softmax(y)
	"logsumexp": (
		_linear,
		lambda y: torch.logsumexp(y, dim=-1),
		lambda x: A.T @ _softmax(_linear(x)),
		lambda x: A.T
		@ (torch.diag(_softmax(_linear(x))) - torch.outer(_softmax(_linear(x)), _softmax(_linear(x))))
		@ A,
	),
}

def _build(case):
	fn, g, grad, hess = CASES[case]
	n_outputs = len(fn(X0))
	return AnalyticModel(fn, X0, n_outputs), fn, g, grad, hess


@pytest.mark.parametrize("case", CASES)
def test_jacobian_and_hessian_match_analytic(case):
	''' Test that the Jacobian and Hessian computed by generalized_taylor match the analytic expressions. '''
	model, _, g, grad, hess = _build(case)
	J, H, _, _ = generalized_taylor(model, model.input_names, model.output_names, g)
	torch.testing.assert_close(J, grad(X0))
	torch.testing.assert_close(H, hess(X0))


@pytest.mark.parametrize("case", CASES)
def test_returns_expansion_point_and_restores_model(case):
	''' Test that generalized_taylor returns the correct expansion point and restores the model state. '''
	model, fn, g, _, _ = _build(case)
	J, H, x0, y0 = generalized_taylor(model, model.input_names, model.output_names, g)

	torch.testing.assert_close(x0, X0)
	torch.testing.assert_close(y0, fn(X0))  # raw model outputs, not g(y0)

	# the model is left at x0 and carries no autograd graph
	state = model.get(model.input_names + model.output_names)
	torch.testing.assert_close(torch.stack([state[k] for k in model.input_names]), X0)
	assert not any(v.requires_grad for v in state.values())
	assert not (J.requires_grad or H.requires_grad)


@pytest.mark.parametrize("case", CASES)
def test_prior_matches_f_at_expansion_point(case):
	''' Test that the prior matches the function value at the expansion point. '''
	model, fn, g, _, _ = _build(case)
	J, H, x0, y0 = generalized_taylor(model, model.input_names, model.output_names, g)
	prior = GenerizalizedTaylorPrior(J, H, x0, g(y0))
	torch.testing.assert_close(prior(X0.unsqueeze(0)), g(fn(X0)).reshape(1))


def test_prior_is_exact_for_quadratic_f():
	''' Test that the prior is exact for a quadratic function. '''
	model, fn, g, _, _ = _build("sum_of_squares")
	J, H, x0, y0 = generalized_taylor(model, model.input_names, model.output_names, g)
	prior = GenerizalizedTaylorPrior(J, H, x0, g(y0))

	x = X0 + torch.randn(20, 3, dtype=DTYPE, generator=torch.Generator().manual_seed(0))
	expected = torch.stack([g(fn(xi)) for xi in x])
	torch.testing.assert_close(prior(x), expected)


@pytest.mark.parametrize("case", ["logsumexp"])
def test_prior_error_is_third_order(case):
	''' Test that the prior error scales as the third power of the step size. '''
	''' Halving the step should shrink |f - prior| by ~8x; a wrong H (e.g. a missing 1/2) gives ~4x. '''
	model, fn, g, _, _ = _build(case)
	J, H, x0, y0 = generalized_taylor(model, model.input_names, model.output_names, g)
	prior = GenerizalizedTaylorPrior(J, H, x0, g(y0))

	direction = torch.tensor([1.0, 0.5, -0.8], dtype=DTYPE)
	steps = [0.1, 0.05, 0.025]
	errors = []
	for t in steps:
		x = (X0 + t * direction).unsqueeze(0)
		errors.append((prior(x) - g(fn(x[0]))).abs().item())

	orders = [math.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]
	assert all(2.6 < p < 3.4 for p in orders), orders


@pytest.mark.parametrize("batch_shape", [(5,), (2, 5), (4, 1, 5)])
def test_prior_batch_shapes(batch_shape):
	''' Test that the prior correctly handles batched inputs and preserves batch shapes. '''
	model, _, g, _, _ = _build("logsumexp")
	J, H, x0, y0 = generalized_taylor(model, model.input_names, model.output_names, g)
	prior = GenerizalizedTaylorPrior(J, H, x0, g(y0))

	x = X0 + 0.1 * torch.randn(*batch_shape, 3, dtype=DTYPE)
	out = prior(x)
	assert out.shape == batch_shape
	torch.testing.assert_close(out.reshape(-1), prior(x.reshape(-1, 3)))
