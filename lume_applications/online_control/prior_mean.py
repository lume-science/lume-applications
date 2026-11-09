import torch
from typing import Callable, Dict,Optional
from lume.model import LUMEModel
from copy import deepcopy 
def generalized_taylor(model: LUMEModel, input_names: list[str], output_names: list[str], g: Callable[[Dict[str,torch.Tensor]], torch.Tensor], x0: Optional[Dict[str,torch.Tensor]]=None):
    """Compute the gradient and Hessian of the scalar-valued function f(x) = g(model(x)) at a specified expansion point.
    The model must preserve the PyTorch computation graph between the input tensors passed to model.set()
    and the output tensors returned by model.get(). The model is deepcopy'd
    so no changes to the original model occur.

    Parameters
    ----------
    model : LUMEModel
        Differentiable LUME model with lume-variables.
        This model is unmodified by any calls to generalized_taylor
        Serves as the reference model for the Taylor expansion.
    input_names : list of str
        Input variables names gather from the model to construct x to differentiate with respect to. Their order fixes the order of the
        entries of x0, J and H.
    output_names : list of str
        Outputs read from the model and passed to g as a dict keyed by name.
    g : callable
        Maps y: dict[str, torch.Tensor] to the scalar f being approximated, e.g.
        g(y) = y["sigma_x"]**2 + y["sigma_y"]**2, Must return a scalar,
        since H is the Hessian of a scalar function.
    x0 : dict[str, torch.Tensor], optional
        Expansion point for the Taylor series. If not None, the model is set
        to these settings and used as the expansion point x0 for the
        Taylor expansion. If None, the model's current inputs are used.

    Returns
    -------
    x0 : torch.Tensor, shape (n,)
        Expansion point.
    f0 : torch.Tensor, scalar
        f at x0, i.e. g(y0) where y0 are the model outputs at x0.
    J : torch.Tensor, shape (n,)
        Gradient of f at x0.
    H : torch.Tensor, shape (n, n)
        Hessian of f at x0.
    """
    model = deepcopy(model)

    if x0 is None:
        init = model.get(input_names)
        x0 = torch.stack([init[k] for k in input_names]).detach()
    else:
        x0 = torch.stack([x0[k] for k in input_names]).detach()
        model.set({k: x0[i] for i, k in enumerate(input_names)})

    y0 = {
        k: v.detach()
        for k, v in model.get(output_names).items()
    }

    def forward(x):
        model.set({k: x[i] for i, k in enumerate(input_names)})
        y = model.get(output_names)
        return g(y)

    J = torch.autograd.functional.jacobian(forward, x0)
    H = torch.autograd.functional.hessian(forward, x0)
    f0 = g(y0).detach()

    return x0, f0, J.detach(), H.detach()

class GeneralizedTaylorPrior(torch.nn.Module):
    """GP prior mean from a second-order Taylor expansion of f about z0:

        f_prior(x) = f0 + J (x - z0) + 1/2 (x - z0)^T H (x - z0)

    The arguments are the outputs of generalized_taylor. They are stored as float64
    buffers, so the prior has no trainable parameters.

    Parameters
    ----------
    z0 : torch.Tensor, shape (n,)
        Expansion point (x0), with inputs in the same order as J and H.
    f0 : torch.Tensor, scalar
        f at z0, i.e. g(y0).
    jacobian : torch.Tensor, shape (n,)
        Gradient of f at z0 (J).
    hessian : Optional[torch.Tensor], shape (n, n)
        Hessian of f at z0 (H).

    """

    def __init__(
        self,
        z0: torch.Tensor,
        f0: torch.Tensor,
        jacobian: torch.Tensor,
        hessian: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.register_buffer("z0", z0.to(torch.float64))
        self.register_buffer("f0", f0.to(torch.float64))
        self.register_buffer("J", jacobian.to(torch.float64))
        self.register_buffer(
            "H",
            None if hessian is None else hessian.to(torch.float64)
)
    def forward(self, x):
        """x: (..., n) inputs, ordered as input_names -> (...,) prior mean f_prior(x)."""
        delta = x - self.z0
        linear_term = delta @ self.J.T
        if self.H is not None:
            quadratic_term = 0.5 * torch.einsum('...i,ij,...j->...', delta, self.H, delta)
        else:
            quadratic_term = 0.0
        return self.f0 + linear_term + quadratic_term

