import torch
from typing import Callable, Dict
from lume.model import LUMEModel

def generalized_taylor(model: LUMEModel, input_names: Dict[str, torch.Tensor], output_names: Dict[str, torch.Tensor], g: Callable[[torch.Tensor], torch.Tensor]):
    """Second-order Taylor expansion of f(x) = g(model(x)) about the model's current inputs.

    J and H come from autograd through model.set/model.get, so nothing between the inputs
    and outputs may break the graph (no .detach(), .item(), or no_grad). The model is set
    repeatedly during the call and is left back at x0.

    Parameters
    ----------
    model : LUMEModel
        Differentiable model. Its current input values are the expansion point x0.
    input_names : list of str
        Inputs x to differentiate with respect to. Their order fixes the order of the
        entries of x0, J and H.
    output_names : list of str
        Outputs stacked, in this order, into the vector y that is passed to g.
    g : callable
        Maps y, shape (m,), to the scalar f being approximated, e.g.
        g(y) = y[0]**2 + y[1]**2 for sigma_x^2 + sigma_y^2. Must return a scalar,
        since H is the Hessian of a scalar function.

    Returns
    -------
    J : torch.Tensor, shape (n,)
        Gradient of f at x0.
    H : torch.Tensor, shape (n, n)
        Hessian of f at x0.
    x0 : torch.Tensor, shape (n,)
        Expansion point.
    y0 : torch.Tensor, shape (m,)
        Model outputs at x0, before g. The prior's f0 is g(y0), not y0.
    """
    init = model.get(input_names)
    x0 = torch.stack([init[k] for k in input_names]).detach()
    out0 = model.get(output_names)
    y0 = torch.stack([out0[k] for k in output_names]).detach()

    def forward(x):
        model.set({k: x[i] for i, k in enumerate(input_names)})
        vals = model.get(output_names)
        y = torch.stack([vals[k] for k in output_names])
        return g(y)

    J = torch.autograd.functional.jacobian(forward, x0)
    H = torch.autograd.functional.hessian(forward, x0)
    model.set({k: x0[i] for i, k in enumerate(input_names)})  # leave the model at x0, without the autograd graph
    return J.detach(), H.detach(), x0, y0

class GenerizalizedTaylorPrior(torch.nn.Module):
    """GP prior mean from a second-order Taylor expansion of f about z0:

        f_prior(x) = f0 + J (x - z0) + 1/2 (x - z0)^T H (x - z0)

    The arguments are the outputs of generalized_taylor. They are stored as float64
    buffers, so the prior has no trainable parameters.

    Parameters
    ----------
    jacobian : torch.Tensor, shape (n,)
        Gradient of f at z0 (J).
    hessian : torch.Tensor, shape (n, n)
        Hessian of f at z0 (H).
    z0 : torch.Tensor, shape (n,)
        Expansion point (x0), with inputs in the same order as J and H.
    f0 : torch.Tensor, scalar
        f at z0, i.e. g(y0).
    """

    def __init__(self, jacobian, hessian, z0, f0):
        super().__init__()
        self.register_buffer("J", jacobian.to(torch.float64))
        self.register_buffer("H", hessian.to(torch.float64))
        self.register_buffer("z0", z0.to(torch.float64))
        self.register_buffer("f0", f0.to(torch.float64))
    def forward(self, x):
        """x: (..., n) inputs, ordered as input_names -> (...,) prior mean f_prior(x)."""
        delta = x - self.z0
        linear_term = delta @ self.J.T
        quadratic_term = 0.5 * torch.einsum('...i,ij,...j->...', delta, self.H, delta)
        return self.f0 + linear_term + quadratic_term

